from typing import Optional, Callable, Any
from observability.logger import get_logger
from config.settings import settings
import time
import functools

logger = get_logger(__name__)


def with_retry(
    max_retries: int = None,
    delay_seconds: float = 2.0,
    fallback: Any = None,
):
    """
    Decorator for retrying flaky operations.
    Used on tool calls and LLM calls that may transiently fail.

    Usage:
        @with_retry(max_retries=3, fallback="Search unavailable")
        def my_tool():
            ...
    """
    max_retries = max_retries or settings.max_retries

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} failed "
                        f"for {func.__name__}: {e}"
                    )
                    if attempt < max_retries:
                        time.sleep(delay_seconds * attempt)

            logger.error(
                f"All {max_retries} attempts failed for "
                f"{func.__name__}: {last_error}"
            )
            if fallback is not None:
                return fallback
            raise last_error
        return wrapper
    return decorator


class WorkflowErrorHandler:
    """
    Centralized error handling for the LangGraph workflow.
    Catches node failures and decides whether to retry or fail gracefully.
    """

    def __init__(self, max_node_retries: int = 2):
        self.max_node_retries = max_node_retries
        self.node_errors: dict = {}

    def handle_node_error(
        self,
        node_name: str,
        error: Exception,
        state: dict,
    ) -> dict:
        """
        Called when a node raises an exception.
        Logs the error, updates state, and returns a safe state.
        """
        error_msg = f"{node_name}: {type(error).__name__}: {str(error)}"
        logger.error(f"Node error — {error_msg}")

        state["errors_encountered"].append(error_msg)
        self.node_errors[node_name] = self.node_errors.get(node_name, 0) + 1

        if self.node_errors[node_name] >= self.max_node_retries:
            logger.error(
                f"Node {node_name} failed {self.max_node_retries} times. "
                f"Marking workflow as degraded."
            )
            state["workflow_status"] = "degraded"

        return state

    def is_recoverable(self, error: Exception) -> bool:
        """
        Determine if an error is transient (retry) or permanent (fail).
        """
        transient_errors = [
            "ConnectionError",
            "TimeoutError",
            "RateLimitError",
            "503",
            "502",
        ]
        error_str = str(type(error).__name__) + str(error)
        return any(e in error_str for e in transient_errors)
    

import time
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"       # normal operation
    OPEN = "open"           # failing — reject calls
    HALF_OPEN = "half_open" # testing recovery


class CircuitBreaker:
    """
    Circuit breaker for external tool calls.
    Prevents cascading failures when external services are down.

    States:
        CLOSED   → calls go through normally
        OPEN     → calls rejected immediately (fail fast)
        HALF_OPEN → one test call allowed to check recovery
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception(
                    f"Circuit breaker OPEN for {self.name}. "
                    f"Try again in {self.recovery_timeout}s."
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    @property
    def is_available(self) -> bool:
        return self.state != CircuitState.OPEN


# Module-level circuit breakers — one per external service
web_search_breaker = CircuitBreaker("web_search", failure_threshold=3, recovery_timeout=120)