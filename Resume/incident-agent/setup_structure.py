import os

folders = [
    "agents",
    "workflows",
    "tools",
    "memory",
    "knowledge_base/chroma_store",
    "knowledge_base/documents",
    "crews",
    "mcp",
    "evaluation/benchmarks",
    "observability",
    "config",
    "data/sample_incidents",
    "data/knowledge_docs",
    "tests/unit",
    "tests/integration",
    "tests/e2e",
    "notebooks",
    "docs",
    "logs",
]

files = [
    # Agents
    "agents/__init__.py",
    "agents/coordinator_agent.py",
    "agents/knowledge_agent.py",
    "agents/resolution_agent.py",

    # Workflows
    "workflows/__init__.py",
    "workflows/state.py",
    "workflows/graph.py",
    "workflows/router.py",

    # Tools
    "tools/__init__.py",
    "tools/search_tool.py",
    "tools/knowledge_tool.py",
    "tools/incident_tool.py",
    "tools/log_parser_tool.py",
    "tools/mcp_tool.py",

    # Memory
    "memory/__init__.py",
    "memory/session_memory.py",
    "memory/historical_memory.py",

    # Knowledge base
    "knowledge_base/__init__.py",
    "knowledge_base/ingest.py",
    "knowledge_base/retriever.py",

    # Crews
    "crews/__init__.py",
    "crews/investigation_crew.py",
    "crews/tasks.py",

    # MCP
    "mcp/__init__.py",
    "mcp/server.py",
    "mcp/handlers.py",

    # Evaluation
    "evaluation/__init__.py",
    "evaluation/metrics.py",
    "evaluation/evaluator.py",

    # Observability
    "observability/__init__.py",
    "observability/logger.py",
    "observability/tracer.py",
    "observability/cost_tracker.py",

    # Config
    "config/__init__.py",
    "config/settings.py",
    "config/models.py",

    # Tests
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "tests/e2e/__init__.py",

    # Root files
    "main.py",
    "README.md",
]

print("\n── Creating project structure ──\n")

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"  [DIR]  {folder}/")

for file in files:
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass
        print(f"  [FILE] {file}")
    else:
        print(f"  [SKIP] {file} already exists")

print("\n── Structure created successfully ──\n")