"""
genai/feedback_generator.py
----------------------------
GenAI layer: generates personalized student feedback using a pluggable LLM backend.

What the LLM is doing here (important distinction):
  - It is NOT being trained. Not even fine-tuned.
  - It is a pre-trained model used as a structured text formatter.
  - Input  → ML prediction (score) + feature attribution (SHAP/gradient values)
  - Output → Natural language coaching paragraph for the student
  - Pattern: "Augmented Generation" — grounding a pre-trained LLM in your data

Supported backends (all FREE options included):
  ┌─────────────────┬────────────────────────┬────────────┬────────────┐
  │ Backend         │ Model                  │ Cost       │ Setup      │
  ├─────────────────┼────────────────────────┼────────────┼────────────┤
  │ Ollama (LOCAL)  │ Llama 3.1, Gemma 2     │ Free       │ Easy       │
  │ Groq            │ Llama 3.1 70B          │ Free tier  │ API key    │
  │ HuggingFace     │ Mistral, Zephyr, etc.  │ Free tier  │ API key    │
  │ Gemini          │ Gemini 1.5 Flash       │ Free tier  │ API key    │
  │ Anthropic       │ Claude Sonnet          │ Paid       │ API key    │
  └─────────────────┴────────────────────────┴────────────┴────────────┘

Set LLM_BACKEND in .env to switch. Zero code changes required.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


# ======================================================================
# System prompt — shared across all backends
# ======================================================================

SYSTEM_PROMPT = """You are an empathetic, experienced academic counselor at a university.
You receive structured data about a student's predicted exam performance and the specific
factors driving that prediction.

Your job is to write a personalized, actionable feedback report for the student.

Guidelines:
- Be warm, encouraging, and constructive — never discouraging
- Translate technical feature names into plain English
- Give 2-3 specific, actionable recommendations
- Acknowledge strengths before addressing weaknesses
- Keep the tone professional but approachable
- Length: 150-250 words
- Do NOT mention "SHAP values", "machine learning", or "model" — speak as a counselor
- Format: plain paragraphs only, no bullet points or headers
"""


# ======================================================================
# Abstract base class — every backend implements one method
# ======================================================================

class LLMBackend(ABC):
    """Interface that every LLM backend must implement."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt, return the text response."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name for logging."""
        ...


# ======================================================================
# Backend: Ollama (RECOMMENDED — 100% local, free, no API key)
# ======================================================================

class OllamaBackend(LLMBackend):
    """
    Calls a locally running Ollama server.

    Setup (one-time, ~5 minutes):
      1. Install:   https://ollama.ai  (Mac/Linux/Windows)
      2. Pull model: ollama pull llama3.1        (4.7 GB, best quality)
                  or ollama pull gemma2:2b        (1.6 GB, fast/lightweight)
                  or ollama pull mistral           (4.1 GB, good balance)
      3. Ollama auto-runs at http://localhost:11434

    Why Ollama for this learning project:
      - Same model families used in LoRA fine-tuning workflows
      - No rate limits, no data sent externally, no cost
      - Experiment with quantization (Q4, Q8 variants)
      - Direct bridge to future fine-tuning practice
    """

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self._check_connection()

    @property
    def name(self) -> str:
        return f"Ollama ({self.model})"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 400},
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _check_connection(self) -> None:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            logger.info(f"Ollama connected. Available models: {models}")
            if not any(self.model.split(":")[0] in m for m in models):
                logger.warning(
                    f"Model '{self.model}' not found locally.\n"
                    f"Run: ollama pull {self.model}"
                )
        except requests.exceptions.ConnectionError:
            logger.warning(
                "Ollama server not running.\n"
                "Start with: ollama serve\n"
                "Install from: https://ollama.ai"
            )


# ======================================================================
# Backend: Groq (free cloud, Llama 3.1 70B, very fast)
# ======================================================================

class GroqBackend(LLMBackend):
    """
    Groq cloud API — free tier with fast inference.

    Setup:
      1. Sign up free: https://console.groq.com
      2. Create API key
      3. Set GROQ_API_KEY in .env

    Free models:
      llama-3.1-70b-versatile   (best quality)
      llama-3.1-8b-instant      (fastest)
      mixtral-8x7b-32768        (long context)
      gemma2-9b-it              (Google model)
    """

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str = "llama-3.1-70b-versatile"):
        self.model = model
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "Get a free key at: https://console.groq.com"
            )

    @property
    def name(self) -> str:
        return f"Groq ({self.model})"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens": 400,
            "temperature": 0.7,
        }
        resp = requests.post(self.API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ======================================================================
# Backend: HuggingFace Inference API (free tier)
# ======================================================================

class HuggingFaceBackend(LLMBackend):
    """
    HuggingFace hosted Inference API — free tier.

    Setup:
      1. Sign up free: https://huggingface.co
      2. Settings → Access Tokens → New Token (read)
      3. Set HF_API_KEY in .env

    Good free models:
      mistralai/Mistral-7B-Instruct-v0.3
      HuggingFaceH4/zephyr-7b-beta
      google/gemma-2-9b-it

    Note: Free tier has rate limits and queue times.
    Use Ollama or Groq for faster iteration during development.
    """

    API_URL = "https://api-inference.huggingface.co/models/{model}"

    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.model = model
        self.api_key = os.environ.get("HF_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "HF_API_KEY not set.\n"
                "Get a free token at: https://huggingface.co/settings/tokens"
            )

    @property
    def name(self) -> str:
        return f"HuggingFace ({self.model})"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        full_prompt = f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 400,
                "temperature": 0.7,
                "return_full_text": False,
            },
        }
        resp = requests.post(
            self.API_URL.format(model=self.model),
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list):
            return result[0].get("generated_text", "").strip()
        return str(result).strip()


# ======================================================================
# Backend: Google Gemini (free tier)
# ======================================================================

class GeminiBackend(LLMBackend):
    """
    Google Gemini API — free tier (Gemini 1.5 Flash).

    Setup:
      1. Go to: https://aistudio.google.com
      2. Click "Get API Key"
      3. Set GEMINI_API_KEY in .env
    """

    API_URL = (
        "https://generativelanguage.googleapis.com/v1beta/models"
        "/{model}:generateContent?key={key}"
    )

    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not set.\n"
                "Get a free key at: https://aistudio.google.com"
            )

    @property
    def name(self) -> str:
        return f"Gemini ({self.model})"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        url = self.API_URL.format(model=self.model, key=self.api_key)
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"maxOutputTokens": 400, "temperature": 0.7},
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return (
            resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        )


# ======================================================================
# Backend: Anthropic Claude (paid, kept for teams with API access)
# ======================================================================

class AnthropicBackend(LLMBackend):
    """
    Claude API. Requires: pip install anthropic && ANTHROPIC_API_KEY in .env
    Highest quality output but not free.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic as _anthropic
            self.client = _anthropic.Anthropic()
        except ImportError:
            raise ImportError("Run: pip install anthropic")
        self.model = model

    @property
    def name(self) -> str:
        return f"Anthropic ({self.model})"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text.strip()


# ======================================================================
# Factory: reads LLM_BACKEND env var and returns the right backend
# ======================================================================

def create_backend(backend: Optional[str] = None) -> LLMBackend:
    """
    Select and instantiate an LLM backend.

    Configure in .env:
      LLM_BACKEND=ollama        # local, free, RECOMMENDED
      LLM_BACKEND=groq          # free cloud, fastest
      LLM_BACKEND=huggingface   # free cloud, most model choice
      LLM_BACKEND=gemini        # free cloud, Google
      LLM_BACKEND=anthropic     # paid Claude

    Per-backend env vars:
      Ollama:       OLLAMA_MODEL (default: llama3.1)
      Groq:         GROQ_API_KEY, GROQ_MODEL
      HuggingFace:  HF_API_KEY, HF_MODEL
      Gemini:       GEMINI_API_KEY
      Anthropic:    ANTHROPIC_API_KEY
    """
    name = (backend or os.environ.get("LLM_BACKEND", "ollama")).lower().strip()
    logger.info(f"Initializing LLM backend: '{name}'")

    if name == "ollama":
        return OllamaBackend(model=os.environ.get("OLLAMA_MODEL", "llama3.1"))
    elif name == "groq":
        return GroqBackend(model=os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile"))
    elif name == "huggingface":
        return HuggingFaceBackend(model=os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"))
    elif name == "gemini":
        return GeminiBackend()
    elif name == "anthropic":
        return AnthropicBackend()
    else:
        raise ValueError(
            f"Unknown backend: '{name}'. "
            f"Valid options: ollama, groq, huggingface, gemini, anthropic"
        )


# ======================================================================
# Main class: StudentFeedbackGenerator
# ======================================================================

class StudentFeedbackGenerator:
    """
    Generates personalized academic feedback using any configured LLM backend.

    The backend is fully swappable — this class has zero backend-specific code.

    Parameters
    ----------
    backend : LLMBackend instance, or None (auto-detected from LLM_BACKEND env var)
    """

    def __init__(self, backend: Optional[LLMBackend] = None):
        self.backend = backend or create_backend()
        logger.info(f"FeedbackGenerator ready — backend: {self.backend.name}")

    def generate(
        self,
        student_data: Dict[str, float],
        predicted_score: float,
        shap_explanation: Dict[str, float],
        student_name: Optional[str] = None,
    ) -> str:
        """
        Generate a personalized feedback report for one student.

        Parameters
        ----------
        student_data      : Raw (unscaled) feature values — interpretable numbers.
        predicted_score   : Model's predicted final exam score (0-100).
        shap_explanation  : {feature_label: contribution_value} from explainability module.
        student_name      : Optional name for personalization.

        Returns
        -------
        Feedback report as plain text.
        """
        prompt = self._build_prompt(
            student_data, predicted_score, shap_explanation, student_name
        )
        logger.info(
            f"Generating feedback via {self.backend.name} "
            f"(predicted score: {predicted_score:.1f})"
        )
        feedback = self.backend.complete(SYSTEM_PROMPT, prompt)
        logger.info(f"Feedback generated ({len(feedback.split())} words).")
        return feedback

    def generate_batch(self, students: list) -> list:
        """
        Generate feedback for a list of students.

        Each element should be a dict with keys matching generate() parameters.
        """
        results = []
        for i, s in enumerate(students):
            logger.info(f"Student {i + 1}/{len(students)}")
            feedback = self.generate(**s)
            results.append({**s, "feedback": feedback})
        return results

    # ------------------------------------------------------------------
    # Internal: prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        student_data: Dict[str, float],
        predicted_score: float,
        shap_explanation: Dict[str, float],
        student_name: Optional[str],
    ) -> str:
        """
        Build the structured context prompt that grounds the LLM in ML findings.

        This is the bridge between structured ML output and free-form LLM generation.
        The quality of this prompt determines the quality of feedback — same principle
        as prompt engineering for any production LLM application.
        """
        name_line = f"Student: {student_name}\n" if student_name else ""

        if predicted_score >= 75:
            risk = "low risk — on track"
        elif predicted_score >= 55:
            risk = "moderate risk — needs improvement"
        else:
            risk = "high risk — urgent intervention needed"

        positive = [
            f"  + {feat} (impact: +{val:.1f} pts)"
            for feat, val in shap_explanation.items() if val > 0
        ]
        negative = [
            f"  - {feat} (impact: {val:.1f} pts)"
            for feat, val in shap_explanation.items() if val < 0
        ]

        prompt = f"""{name_line}
STUDENT PERFORMANCE ANALYSIS
=============================
Predicted Final Exam Score : {predicted_score:.1f} / 100
Risk Category              : {risk}

CURRENT SEMESTER METRICS:
  Weekly study hours        : {student_data.get('hours_studied_per_week', 'N/A'):.1f} hrs
  Class attendance          : {student_data.get('attendance_percentage', 'N/A'):.1f}%
  Assignment completion     : {student_data.get('assignments_completion_rate', 'N/A'):.1f}%
  Previous exam score       : {student_data.get('previous_exam_score', 'N/A'):.1f} / 100
  Average sleep per night   : {student_data.get('sleep_hours_per_night', 'N/A'):.1f} hrs

FACTORS HELPING THIS STUDENT:
{chr(10).join(positive) if positive else '  (no significant positive factors)'}

FACTORS HURTING THIS STUDENT:
{chr(10).join(negative) if negative else '  (no significant negative factors)'}

Please write a personalized counseling report for this student.
""".strip()
        return prompt


# ======================================================================
# CLI demo
# ======================================================================

if __name__ == "__main__":
    """
    Quick demo. Configure LLM_BACKEND env var before running.

    Examples:
      LLM_BACKEND=ollama python -m genai.feedback_generator
      LLM_BACKEND=groq   python -m genai.feedback_generator
    """
    import logging
    logging.basicConfig(level=logging.INFO)

    student = {
        "hours_studied_per_week": 12.0,
        "attendance_percentage": 65.0,
        "assignments_completion_rate": 70.0,
        "previous_exam_score": 55.0,
        "sleep_hours_per_night": 5.5,
    }
    shap_explanation = {
        "Weekly Study Hours": -4.2,
        "Class Attendance": -6.1,
        "Assignment Completion": -2.3,
        "Previous Exam Score": -8.0,
        "Sleep Hours": -5.5,
    }

    gen = StudentFeedbackGenerator()
    feedback = gen.generate(
        student_data=student,
        predicted_score=52.3,
        shap_explanation=shap_explanation,
        student_name="Alex",
    )

    print("\n" + "=" * 60)
    print(f"BACKEND : {gen.backend.name}")
    print("=" * 60)
    print(feedback)
