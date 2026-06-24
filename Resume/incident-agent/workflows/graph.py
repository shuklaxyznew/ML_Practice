from langgraph.graph import StateGraph, END
from workflows.state import AgentState, create_initial_state
from workflows.router import route_after_coordinator, route_after_reflection
from observability.logger import get_logger
from config.settings import settings
import time

logger = get_logger(__name__)


def build_graph():
    from tools.search_tool import web_search_tool
    from tools.knowledge_tool import knowledge_retrieval_tool
    from tools.incident_tool import incident_lookup_tool, find_similar_incidents_tool
    from tools.log_parser_tool import log_parser_tool
    from agents.coordinator_agent import create_coordinator_agent
    from agents.knowledge_agent import create_knowledge_agent
    from agents.resolution_agent import create_resolution_agent

    # Tool assignments per agent
    coordinator_tools = [
        web_search_tool,
        incident_lookup_tool,
    ]
    knowledge_tools = [
        web_search_tool,
        knowledge_retrieval_tool,
        find_similar_incidents_tool,
        log_parser_tool,
    ]
    resolution_tools = [
        knowledge_retrieval_tool,
        find_similar_incidents_tool,
        log_parser_tool,
    ]

    coordinator = create_coordinator_agent(coordinator_tools)
    knowledge = create_knowledge_agent(knowledge_tools)
    resolution = create_resolution_agent(resolution_tools)

    # ── Node functions ──

    def intake_node(state: AgentState) -> AgentState:
        logger.info(f"INTAKE: {state['incident']['incident_id']}")
        state["current_node"] = "intake"
        state["needs_more_info"] = True
        return state

    def coordinator_node(state: AgentState) -> AgentState:
        logger.info("COORDINATOR: planning investigation")
        state["current_node"] = "coordinator"
        state["iteration_count"] += 1

        incident = state["incident"]
        input_text = (
            f"Incident ID: {incident['incident_id']}\n"
            f"Title: {incident['title']}\n"
            f"Description: {incident['description']}\n"
            f"Affected Service: {incident['affected_service']}\n"
            f"Severity: {incident['severity']}"
        )

        try:
            result = coordinator.invoke({"input": input_text})
            output = result.get("output", "")
            state["knowledge_context"].append(f"Coordinator plan:\n{output}")
            state["tool_calls_made"].extend([
                step[0].tool for step in result.get("intermediate_steps", [])
            ])
        except Exception as e:
            logger.error(f"Coordinator error: {e}")
            state["errors_encountered"].append(str(e))

        return state

    def knowledge_node(state: AgentState) -> AgentState:
        logger.info("KNOWLEDGE: gathering context")
        state["current_node"] = "knowledge_agent"

        incident = state["incident"]
        context_so_far = "\n".join(state.get("knowledge_context", []))

        input_text = (
            f"Investigate this incident:\n"
            f"Title: {incident['title']}\n"
            f"Description: {incident['description']}\n"
            f"Service: {incident['affected_service']}\n"
            f"Logs: {incident.get('raw_logs', 'None provided')}\n\n"
            f"Context gathered so far:\n{context_so_far}"
        )

        try:
            result = knowledge.invoke({"input": input_text})
            output = result.get("output", "")
            state["knowledge_context"].append(f"Knowledge findings:\n{output}")
            state["tool_calls_made"].extend([
                step[0].tool for step in result.get("intermediate_steps", [])
            ])
        except Exception as e:
            logger.error(f"Knowledge agent error: {e}")
            state["errors_encountered"].append(str(e))

        return state

    def resolution_node(state: AgentState) -> AgentState:
        logger.info("RESOLUTION: analyzing and generating recommendations")
        state["current_node"] = "resolution_agent"

        incident = state["incident"]
        context = "\n\n".join(state.get("knowledge_context", []))
        search = "\n\n".join(state.get("search_results", []))

        input_text = (
            f"Incident: {incident['title']}\n"
            f"Service: {incident['affected_service']}\n"
            f"Description: {incident['description']}\n\n"
            f"Gathered Context:\n{context}\n\n"
            f"Search Results:\n{search}"
        )

        try:
            result = resolution.invoke({"input": input_text})
            output = result.get("output", "")
            state["root_cause"] = output
            state["recommendations"].append(output)
            state["tool_calls_made"].extend([
                step[0].tool for step in result.get("intermediate_steps", [])
            ])
        except Exception as e:
            logger.error(f"Resolution agent error: {e}")
            state["errors_encountered"].append(str(e))

        return state

    def reflection_node(state: AgentState) -> AgentState:
        logger.info("REFLECTION: evaluating output quality")
        state["current_node"] = "reflection"

        root_cause = state.get("root_cause", "")
        recommendations = state.get("recommendations", [])
        errors = state.get("errors_encountered", [])

        score = 0.5

        if root_cause and len(root_cause) > 100:
            score += 0.2
        if recommendations:
            score += 0.15
        if not errors:
            score += 0.1
        if state.get("knowledge_context"):
            score += 0.05

        score = min(score, 1.0)
        state["confidence_score"] = score

        note = (
            f"Confidence: {score:.2f}. "
            f"Root cause identified: {bool(root_cause)}. "
            f"Recommendations: {len(recommendations)}. "
            f"Errors: {len(errors)}."
        )
        state["reflection_notes"].append(note)
        state["needs_more_info"] = score < settings.confidence_threshold

        logger.info(f"Reflection score: {score:.2f}")
        return state

    def output_node(state: AgentState) -> AgentState:
        logger.info("OUTPUT: generating final report")
        state["current_node"] = "output"
        state["workflow_status"] = "complete"

        incident = state["incident"]
        elapsed = time.time() - state["start_time"]

        report = f"""
╔══════════════════════════════════════════════════════════════╗
  INCIDENT RESOLUTION REPORT
  Generated by Enterprise Incident Resolution Agent
╚══════════════════════════════════════════════════════════════╝

INCIDENT DETAILS
  ID:       {incident['incident_id']}
  Title:    {incident['title']}
  Service:  {incident['affected_service']}
  Severity: {incident['severity']}

ANALYSIS
{state.get('root_cause', 'Analysis not available')}

RECOMMENDATIONS
{chr(10).join(f'  • {r}' for r in state.get('recommendations', ['No recommendations generated']))}

INVESTIGATION METADATA
  Confidence Score:  {state.get('confidence_score', 0):.0%}
  Iterations:        {state.get('iteration_count', 0)}
  Tools Used:        {', '.join(set(state.get('tool_calls_made', [])))}
  Errors:            {len(state.get('errors_encountered', []))}
  Time Elapsed:      {elapsed:.1f}s
        """.strip()

        state["final_report"] = report
        return state

    # ── Build the graph ──
    graph = StateGraph(AgentState)

    graph.add_node("intake", intake_node)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("knowledge_agent", knowledge_node)
    graph.add_node("resolution_agent", resolution_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("output", output_node)

    graph.set_entry_point("intake")

    graph.add_edge("intake", "coordinator")
    graph.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {"knowledge_agent": "knowledge_agent", "resolution_agent": "resolution_agent"},
    )
    graph.add_edge("knowledge_agent", "resolution_agent")
    graph.add_edge("resolution_agent", "reflection")
    graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {"output": "output", "coordinator": "coordinator"},
    )
    graph.add_edge("output", END)

    return graph.compile()