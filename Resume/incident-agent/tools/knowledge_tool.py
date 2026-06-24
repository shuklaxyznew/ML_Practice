from langchain.tools import tool
from pydantic import BaseModel, Field
from observability.logger import get_logger

logger = get_logger(__name__)


class KnowledgeInput(BaseModel):
    query: str = Field(
        description="Natural language query to search the internal knowledge base"
    )
    n_results: int = Field(
        default=3,
        description="Number of relevant documents to retrieve"
    )


@tool("knowledge_retrieval", args_schema=KnowledgeInput)
def knowledge_retrieval_tool(query: str, n_results: int = 3) -> str:
    """
    Search the internal knowledge base for relevant runbooks, past
    incident reports, SOPs, and technical documentation. Use this
    before web search — prefer internal knowledge first.
    """
    logger.info(f"Knowledge retrieval: {query}")
    try:
        from knowledge_base.retriever import retrieve
        results = retrieve(query, n_results=n_results)

        if not results:
            return "No relevant documents found in knowledge base."

        formatted = []
        for i, doc in enumerate(results, 1):
            formatted.append(
                f"Document {i}:\n"
                f"Content: {doc['content']}\n"
                f"Source: {doc.get('source', 'internal')}\n"
                f"Relevance: {doc.get('distance', 'N/A')}"
            )

        output = "\n\n".join(formatted)
        logger.info(f"Retrieved {len(results)} documents")
        return output

    except Exception as e:
        logger.error(f"Knowledge retrieval failed: {e}")
        return f"Knowledge base unavailable: {str(e)}"