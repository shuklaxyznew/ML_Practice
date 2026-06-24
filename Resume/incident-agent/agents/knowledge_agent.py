from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from typing import List
from observability.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

KNOWLEDGE_PROMPT = PromptTemplate.from_template("""
You are the Knowledge Agent for an enterprise incident resolution platform.
Your role is to gather all relevant information about an incident from
available sources — internal knowledge base, web search, and incident history.

You have access to the following tools:
{tools}

Tool names: {tool_names}

Investigation Task:
{input}

Instructions:
- Always check the internal knowledge base first
- Search for similar past incidents
- Use web search only for information not in internal sources
- Compile all findings into a comprehensive context summary

Use this format strictly:
Thought: [your reasoning]
Action: [tool name]
Action Input: [tool input]
Observation: [tool result]
... (repeat as needed)
Thought: I have gathered sufficient context
Final Answer: [comprehensive context summary with all findings]

{agent_scratchpad}
""")


def create_knowledge_agent(tools: List[BaseTool]) -> AgentExecutor:
    llm = ChatOllama(
        model=settings.knowledge_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature,
    )

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=KNOWLEDGE_PROMPT,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=settings.max_iterations,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    logger.info(f"Knowledge agent ready with {len(tools)} tools")
    return executor