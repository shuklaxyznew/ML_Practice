import sys

def check(label, fn):
    try:
        fn()
        print(f"  [OK]  {label}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False

print("\n── Environment Validation ──\n")

results = []

# Python version
results.append(check(
    f"Python version: {sys.version.split()[0]}",
    lambda: None
))

# Core imports
results.append(check("LangChain", lambda: __import__("langchain")))
results.append(check("LangGraph", lambda: __import__("langgraph")))
results.append(check("LangChain-Ollama", lambda: __import__("langchain_ollama")))
results.append(check("LangChain-Community", lambda: __import__("langchain_community")))
results.append(check("CrewAI", lambda: __import__("crewai")))
results.append(check("ChromaDB", lambda: __import__("chromadb")))
results.append(check("Sentence-Transformers", lambda: __import__("sentence_transformers")))
results.append(check("DuckDuckGo Search", lambda: __import__("duckduckgo_search")))
results.append(check("FastMCP", lambda: __import__("fastmcp")))
results.append(check("Pydantic", lambda: __import__("pydantic")))
results.append(check("Python-dotenv", lambda: __import__("dotenv")))
results.append(check("Rich", lambda: __import__("rich")))
results.append(check("Ollama client", lambda: __import__("ollama")))

# Ollama connectivity
def test_ollama():
    import ollama
    models = ollama.list()
    names = [m.model for m in models.models]
    assert "qwen2.5:3b" in names, f"qwen2.5:3b not found. Available: {names}"
    assert "gemma3:4b" in names, f"gemma3:4b not found. Available: {names}"

results.append(check("Ollama running + models present", test_ollama))

# LLM smoke test
def test_llm():
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="qwen2.5:3b", temperature=0)
    response = llm.invoke("Reply with the single word: ready")
    assert "ready" in response.content.lower(), f"Unexpected: {response.content}"

results.append(check("LLM inference (Qwen 2.5 3B)", test_llm))

# Embedding smoke test
def test_embeddings():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vec = model.encode("test sentence")
    assert len(vec) == 384, f"Unexpected embedding dim: {len(vec)}"

results.append(check("BGE-Small embeddings", test_embeddings))

# ChromaDB smoke test
def test_chroma():
    import chromadb
    client = chromadb.Client()
    col = client.get_or_create_collection("test")
    col.add(documents=["test doc"], ids=["1"])
    res = col.query(query_texts=["test"], n_results=1)
    assert len(res["documents"]) == 1

results.append(check("ChromaDB read/write", test_chroma))

print(f"\n── Result: {sum(results)}/{len(results)} checks passed ──\n")