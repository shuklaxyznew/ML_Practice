from crewai import Agent, Crew, Process
from langchain_ollama import ChatOllama
from crews.tasks import (
    create_investigation_task,
    create_resolution_task,
    create_validation_task,
)
from tools.knowledge_tool import knowledge_retrieval_tool
from tools.incident_tool import find_similar_incidents_tool
from tools.log_parser_tool import log_parser_tool
from tools.search_tool import web_search_tool
from observability.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


def build_investigation_crew(
    incident_title: str,
    incident_description: str,
    affected_service: str,
    raw_logs: str = "",
) -> dict:
    """
    Build and run a CrewAI crew for incident investigation.
    Returns a dict with investigation and resolution outputs.

    This is called FROM LangGraph nodes — CrewAI handles
    the agent collaboration, LangGraph handles the workflow.
    """

    # ── LLM setup ──
    knowledge_llm = ChatOllama(
        model=settings.knowledge_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature,
    )

    resolution_llm = ChatOllama(
        model=settings.resolution_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )

    # ── Crew Agents ──
    # Note: CrewAI agents have roles, goals, and backstories
    # This is what differentiates them from LangChain agents
    knowledge_crew_agent = Agent(
        role="Senior Knowledge Engineer",
        goal=(
            "Gather comprehensive, accurate information about incidents "
            "from all available sources — internal runbooks, past incidents, "
            "and log analysis. Leave no stone unturned."
        ),
        backstory=(
            "You are a veteran knowledge engineer with 10 years of experience "
            "in enterprise incident management. You know exactly where to look "
            "for information and how to connect dots between disparate sources. "
            "You are methodical, thorough, and always cite your sources."
        ),
        tools=[
            knowledge_retrieval_tool,
            find_similar_incidents_tool,
            log_parser_tool,
            web_search_tool,
        ],
        llm=knowledge_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

    resolution_crew_agent = Agent(
        role="Principal Site Reliability Engineer",
        goal=(
            "Analyze investigation findings and produce clear, actionable "
            "incident resolutions with root cause analysis and preventive measures. "
            "Your recommendations must be specific, executable, and prioritized."
        ),
        backstory=(
            "You are a principal SRE with deep expertise in distributed systems, "
            "database engineering, and incident command. You have resolved hundreds "
            "of P1 incidents and know that good RCA requires evidence, not guesses. "
            "You write recommendations that junior engineers can execute immediately."
        ),
        tools=[
            knowledge_retrieval_tool,
            find_similar_incidents_tool,
            log_parser_tool,
        ],
        llm=resolution_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

    # ── Tasks ──
    investigation_task = create_investigation_task(
        incident_title=incident_title,
        incident_description=incident_description,
        affected_service=affected_service,
    )
    investigation_task.agent = knowledge_crew_agent

    resolution_task = create_resolution_task(
        investigation_context=incident_description,
        incident_title=incident_title,
    )
    resolution_task.agent = resolution_crew_agent

    validation_task = create_validation_task(
        resolution=incident_description,
        incident_title=incident_title,
    )
    validation_task.agent = resolution_crew_agent

    # ── Crew ──
    crew = Crew(
        agents=[knowledge_crew_agent, resolution_crew_agent],
        tasks=[investigation_task, resolution_task, validation_task],
        process=Process.sequential,  # tasks run in order
        verbose=True,
    )

    logger.info(f"Running investigation crew for: {incident_title}")

    try:
        result = crew.kickoff()
        logger.info("Crew investigation complete")
        return {
            "status": "complete",
            "crew_output": str(result),
            "tasks_completed": len(crew.tasks),
        }
    except Exception as e:
        logger.error(f"Crew failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "crew_output": "",
        }