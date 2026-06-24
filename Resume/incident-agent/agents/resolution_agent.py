from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from typing import List
from observability.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

RESOLUTION_PROMPT = PromptTemplate.from_template("""
You are the Resolution Agent for an enterprise incident resolution platform.
You are a senior Site Reliability Engineer with deep expertise in distributed
systems, root cause analysis, and incident management.

You have access to ONLY these tools:
{tools}

Available tool names: {tool_names}

Incident Context and Evidence:
{input}

STRICT RULES:
- Use at most 2 tool calls total, then go directly to Final Answer.
- Once you write "Final Answer:", do NOT write any more Thought or Action lines.
- Never mix Final Answer with Action in the same response.
- Your Final Answer must be a complete structured report.

Use this exact format:
Thought: [reasoning]
Action: [tool name]
Action Input: [input]
Observation: [result]
Thought: I have sufficient evidence for a complete analysis
Final Answer:
ROOT CAUSE: [single most likely root cause with evidence]

CONTRIBUTING FACTORS:
- [factor 1]
- [factor 2]

IMMEDIATE ACTIONS:
1. [action 1]
2. [action 2]
3. [action 3]

PREVENTIVE MEASURES:
1. [measure 1]
2. [measure 2]

SEVERITY: [P1/P2/P3/P4] — [justification]

CONFIDENCE: [0.0-1.0] — [reasoning]

{agent_scratchpad}
""")


def create_resolution_agent(tools: List[BaseTool]) -> AgentExecutor:
    llm = ChatOllama(
        model=settings.resolution_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=RESOLUTION_PROMPT,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=settings.max_iterations,
        verbose=True,
        handle_parsing_errors="Check your output format. Provide ONLY a Final Answer, no Action after Final Answer.",
        return_intermediate_steps=True,
        early_stopping_method="generate",  # Add this line
    )

    logger.info(f"Resolution agent ready with {len(tools)} tools")
    return executor