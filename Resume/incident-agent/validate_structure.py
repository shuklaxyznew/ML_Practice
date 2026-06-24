import os
from pathlib import Path

print("\n── Validating project structure ──\n")

required = [
    "agents/coordinator_agent.py",
    "agents/knowledge_agent.py",
    "agents/resolution_agent.py",
    "workflows/state.py",
    "workflows/graph.py",
    "workflows/router.py",
    "tools/search_tool.py",
    "tools/knowledge_tool.py",
    "tools/incident_tool.py",
    "memory/session_memory.py",
    "memory/historical_memory.py",
    "knowledge_base/ingest.py",
    "knowledge_base/retriever.py",
    "crews/investigation_crew.py",
    "mcp/server.py",
    "evaluation/evaluator.py",
    "observability/logger.py",
    "config/settings.py",
    "workflows/state.py",
    "main.py",
    ".env",
    "requirements.txt",
]

all_good = True
for path in required:
    exists = Path(path).exists()
    status = "[OK]  " if exists else "[MISSING]"
    print(f"  {status} {path}")
    if not exists:
        all_good = False

print()

# Validate settings load
try:
    from config.settings import settings
    print(f"  [OK]  Settings loaded")
    print(f"        coordinator_model = {settings.coordinator_model}")
    print(f"        resolution_model  = {settings.resolution_model}")
    print(f"        chroma_persist_dir = {settings.chroma_persist_dir}")
except Exception as e:
    print(f"  [FAIL] Settings: {e}")
    all_good = False

# Validate AgentState
try:
    from workflows.state import create_initial_state
    state = create_initial_state(
        incident_id="INC-TEST-001",
        title="Validation test",
        description="Testing state creation",
        affected_service="test-service",
    )
    print(f"  [OK]  AgentState created")
    print(f"        incident_id = {state['incident']['incident_id']}")
    print(f"        workflow_status = {state['workflow_status']}")
except Exception as e:
    print(f"  [FAIL] AgentState: {e}")
    all_good = False

# Validate logger
try:
    from observability.logger import get_logger
    logger = get_logger("validation")
    logger.info("Logger working correctly")
    print(f"  [OK]  Logger initialized")
except Exception as e:
    print(f"  [FAIL] Logger: {e}")
    all_good = False

print(f"\n── {'All checks passed' if all_good else 'Some checks failed — fix before continuing'} ──\n")