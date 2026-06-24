from workflows.state import AgentState
from observability.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


def route_after_coordinator(state: AgentState) -> str:
    """
    After the coordinator plans the investigation,
    decide whether knowledge gathering is needed.
    """
    if state.get("needs_more_info", True):
        logger.info("Router: coordinator → knowledge_agent")
        return "knowledge_agent"
    logger.info("Router: coordinator → resolution_agent (direct)")
    return "resolution_agent"


def route_after_reflection(state: AgentState) -> str:
    """
    After reflection, decide whether to finalize or retry.
    """
    confidence = state.get("confidence_score", 0.0)
    iterations = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 10)

    if iterations >= max_iter:
        logger.warning(f"Router: max iterations reached ({max_iter}), forcing output")
        return "output"

    if confidence >= settings.confidence_threshold:
        logger.info(f"Router: confidence {confidence:.2f} >= threshold, → output")
        return "output"

    logger.info(f"Router: confidence {confidence:.2f} < threshold, → retry")
    return "coordinator"