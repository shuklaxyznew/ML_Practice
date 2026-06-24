from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from typing import List
from observability.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

COORDINATOR_PROMPT = PromptTemplate.from_template("""
You are the Coordinator Agent for an enterprise incident resolution platform.
Your role is to analyze incoming incidents and create a clear investigation plan.

You have access to ONLY these tools:
{tools}

Available tool names: {tool_names}

RULES:
- Only use tools from the list above. Never invent tool names.
- Use web_search at most ONCE per investigation.
- If search fails, proceed using only available information.
- When you have enough information, go directly to Final Answer.
- Do not repeat the same action twice.

Incident Details:
{input}

Use this exact format:
Thought: [your reasoning]
Action: [tool name from list above]
Action Input: [input string]
Observation: [tool result]
Thought: I have enough information to create an investigation plan
Final Answer: [your investigation plan with severity assessment and recommended next steps]

{agent_scratchpad}
""")


def create_coordinator_agent(tools: List[BaseTool]) -> AgentExecutor:
    llm = ChatOllama(
        model=settings.coordinator_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature,
    )

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=COORDINATOR_PROMPT,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=settings.max_iterations,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    logger.info(f"Coordinator agent ready with {len(tools)} tools")
    return executor