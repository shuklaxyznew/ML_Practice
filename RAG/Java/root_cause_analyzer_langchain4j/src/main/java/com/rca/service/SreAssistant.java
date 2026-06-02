package com.rca.service;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

/**
 * LangChain4j AI Service — declarative LLM interface.
 *
 * This is LangChain4j's equivalent of LangChain (Python) chains/agents.
 * LangChain4j generates a proxy implementation at runtime that:
 *   1. Injects the @SystemMessage as the system prompt
 *   2. Fills @V template variables into @UserMessage
 *   3. Calls the configured ChatLanguageModel (Claude / OpenAI)
 *   4. Returns the response as a plain String
 *
 * When wired with a RetrievalAugmentor (see LangChain4jConfig), every call
 * automatically retrieves relevant documents from the EmbeddingStore (RAG)
 * and injects them into the prompt before the LLM sees it — exactly like
 * LangChain Python's RetrievalQA chain.
 */
public interface SreAssistant {

    @SystemMessage("""
        You are a Senior Site Reliability Engineer with 10+ years of experience debugging production incidents.

        Your task: Analyze the provided logs, metrics, and past incident context to identify the most
        probable root cause of the failure.

        STRICT OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no preamble:

        {
          "root_cause": "One clear, specific sentence naming the root cause",
          "confidence": 85,
          "severity": "critical",
          "error_count": 4,
          "warning_count": 2,
          "affected_services": ["service-name"],
          "mttr_estimate_minutes": 30,
          "evidence": [
            { "text": "Exact short log snippet (max 100 chars)", "type": "error" }
          ],
          "suggested_actions": [
            {
              "step": 1,
              "priority": "immediate",
              "title": "Short action title",
              "detail": "Specific, technical, actionable instruction"
            }
          ]
        }

        Rules:
        - confidence: integer 0-100. Be realistic. Ambiguous logs → 40-60.
        - severity: "critical" | "high" | "medium" | "low"
        - evidence: 2-5 items quoting actual log lines. Max 100 chars each.
        - suggested_actions: 3-5 items. Priority order: immediate → short-term → long-term.
        - Ground ALL suggestions in the log evidence. Do NOT hallucinate fixes.
        - If past incidents from the context are relevant, reference them in your actions.
        """)
    @UserMessage("""
        Service: {{serviceName}}
        Errors: {{errorCount}} | Warnings: {{warningCount}}

        === CURRENT LOGS ===
        {{logSummary}}

        {{metrics}}
        """)
    String analyze(
        @V("serviceName")  String serviceName,
        @V("errorCount")   int    errorCount,
        @V("warningCount") int    warningCount,
        @V("logSummary")   String logSummary,
        @V("metrics")      String metrics
    );
}
