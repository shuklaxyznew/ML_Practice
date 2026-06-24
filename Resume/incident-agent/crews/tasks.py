from crewai import Task


def create_investigation_task(incident_title: str, incident_description: str, affected_service: str) -> Task:
    """
    Task 1 — assigned to the Knowledge Crew Agent.
    Gather all relevant context about the incident.
    """
    return Task(
        description=f"""
        Investigate the following incident thoroughly:

        Title: {incident_title}
        Service: {affected_service}
        Description: {incident_description}

        Your investigation must:
        1. Search internal knowledge base for relevant runbooks
        2. Find similar past incidents and their resolutions
        3. Identify key error patterns and contributing factors
        4. Compile all findings into a structured context report

        Be specific. Reference actual runbook steps.
        Include confidence level for each finding.
        """,
        expected_output="""
        A structured investigation report containing:
        - Relevant runbooks found (with source references)
        - Similar past incidents (with resolution summaries)
        - Key error patterns identified
        - Initial root cause hypothesis
        - Confidence level: LOW / MEDIUM / HIGH
        """,
        async_execution=False,
    )


def create_resolution_task(investigation_context: str, incident_title: str) -> Task:
    """
    Task 2 — assigned to the Resolution Crew Agent.
    Produce actionable resolution based on investigation findings.
    """
    return Task(
        description=f"""
        Based on the following investigation findings, produce a complete
        incident resolution plan.

        Incident: {incident_title}

        Investigation Context:
        {investigation_context}

        Your resolution must include:
        1. Single definitive root cause statement
        2. Contributing factors (secondary causes)
        3. Immediate remediation steps (numbered, actionable)
        4. Preventive measures (long-term fixes)
        5. Severity classification (P1/P2/P3/P4) with justification
        6. Confidence score (0.0 to 1.0) with reasoning

        Reference specific runbook steps where applicable.
        """,
        expected_output="""
        A complete resolution report with:
        - ROOT CAUSE: (single statement)
        - CONTRIBUTING FACTORS: (bullet list)
        - IMMEDIATE ACTIONS: (numbered steps)
        - PREVENTIVE MEASURES: (numbered steps)
        - SEVERITY: P1/P2/P3/P4 with justification
        - CONFIDENCE: 0.0-1.0 with reasoning
        """,
        async_execution=False,
    )


def create_validation_task(resolution: str, incident_title: str) -> Task:
    """
    Task 3 — assigned to the Resolution Crew Agent (self-review).
    Validate the resolution for completeness and accuracy.
    """
    return Task(
        description=f"""
        Review and validate the following incident resolution for quality:

        Incident: {incident_title}

        Resolution to validate:
        {resolution}

        Check for:
        1. Is the root cause specific and evidence-based?
        2. Are immediate actions concrete and executable?
        3. Are preventive measures realistic?
        4. Is the severity classification justified?
        5. Are there any gaps or missing information?

        If gaps found, note them explicitly.
        Provide a final quality score: POOR / ACCEPTABLE / GOOD / EXCELLENT
        """,
        expected_output="""
        Validation report with:
        - Quality score: POOR / ACCEPTABLE / GOOD / EXCELLENT
        - Gaps identified (if any)
        - Recommended improvements (if any)
        - Final verdict: APPROVED / NEEDS_REVISION
        """,
        async_execution=False,
    )