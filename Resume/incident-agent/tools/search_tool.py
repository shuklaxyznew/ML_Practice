from langchain.tools import tool
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
from observability.logger import get_logger
from config.settings import settings
import time

logger = get_logger(__name__)


class SearchInput(BaseModel):
    query: str = Field(
        description="Search query for finding incident-related information, error codes, or technical details"
    )
    max_results: int = Field(
        default=3,
        description="Maximum number of results to return"
    )


_last_search_time = 0
SEARCH_DELAY_SECONDS = 3


@tool("web_search", args_schema=SearchInput)
def web_search_tool(query: str, max_results: int = 3) -> str:
    """
    Search the web for current information about incidents, error codes,
    outages, or technical issues. Use this ONLY when internal knowledge
    base has no relevant information. Do not call this more than once
    per investigation step.
    """
    global _last_search_time

    # Rate limit guard — wait between searches
    elapsed = time.time() - _last_search_time
    if elapsed < SEARCH_DELAY_SECONDS:
        wait = SEARCH_DELAY_SECONDS - elapsed
        logger.info(f"Rate limit guard: waiting {wait:.1f}s before search")
        time.sleep(wait)

    logger.info(f"Web search: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        _last_search_time = time.time()

        if not results:
            return "No results found. Use internal knowledge base instead."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"Result {i}:\n"
                f"Title: {r.get('title', 'N/A')}\n"
                f"Summary: {r.get('body', 'N/A')}\n"
                f"Source: {r.get('href', 'N/A')}"
            )

        output = "\n\n".join(formatted)
        logger.info(f"Search returned {len(results)} results")
        return output

    except Exception as e:
        _last_search_time = time.time()
        logger.error(f"Search failed: {e}")
        return "Search unavailable. Use internal knowledge base and incident history instead."