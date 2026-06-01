"""
genai/feedback_generator.py
----------------------------
GenAI layer: generates personalized student feedback using Claude (Anthropic API).

This is where ML predictions + SHAP explanations become
actionable, human-readable coaching reports.

Architecture:
  ML Model → predicted score
  SHAP     → why this score (feature contributions)
  LLM      → "Here's what you should do about it" in natural language

This is a practical example of the Augmented Generation pattern:
  - Structured data (predictions + SHAP) → context injection → LLM
  - Not RAG, but the same principle: ground the LLM in your data
"""

import os
import json
import logging
from typing import Dict, Optional
import anthropic

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an empathetic, experienced academic counselor at a university.
You receive structured data about a student's predicted exam performance and the specific 
factors driving that prediction (from a machine learning model's SHAP analysis).

Your job is to write a personalized, actionable feedback report for the student.

Guidelines:
- Be warm, encouraging, and constructive — never discouraging
- Translate technical feature names into plain English
- Give 2-3 specific, actionable recommendations
- Acknowledge strengths before addressing weaknesses
- Keep the tone professional but approachable
- Length: 150-250 words
- Do NOT mention "SHAP values", "machine learning", or "model" — speak as a counselor
- Format: plain paragraphs, no bullet points or headers
"""


class StudentFeedbackGenerator:
    """
    Generates personalized academic feedback reports using Claude.

    Parameters
    ----------
    model_name  : Claude model to use. claude-sonnet-4-20250514 recommended.
    max_tokens  : Maximum tokens in the generated feedback.
    """

    def __init__(
        self,
        model_name: str  = "claude-sonnet-4-20250514",
        max_tokens: int  = 400,
    ):
        self.client     = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model_name = model_name
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        student_data      : Raw feature values (unscaled, interpretable).
        predicted_score   : Model's predicted final exam score (0-100).
        shap_explanation  : Dict of {feature_label: shap_value} from ModelExplainer.
        student_name      : Optional student name for personalization.

        Returns
        -------
        Feedback report as a plain-text string.
        """
        prompt = self._build_prompt(
            student_data, predicted_score, shap_explanation, student_name
        )

        logger.info(f"Generating feedback for student (predicted score: {predicted_score:.1f})")

        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        feedback = message.content[0].text
        logger.info(f"Feedback generated ({len(feedback.split())} words)")
        return feedback

    def generate_batch(
        self,
        students: list,
    ) -> list:
        """
        Generate feedback for a list of students.

        Each item in `students` should be a dict with keys:
          - student_data, predicted_score, shap_explanation, student_name (optional)
        """
        results = []
        for i, s in enumerate(students):
            logger.info(f"Processing student {i+1}/{len(students)}")
            feedback = self.generate(**s)
            results.append({
                **s,
                "feedback": feedback,
            })
        return results

    # ------------------------------------------------------------------
    # Internal: Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        student_data: Dict[str, float],
        predicted_score: float,
        shap_explanation: Dict[str, float],
        student_name: Optional[str],
    ) -> str:
        """
        Build a structured context prompt that grounds the LLM
        in the ML model's findings.

        This is the bridge between structured ML output and free-form LLM generation.
        """
        name_line = f"Student: {student_name}\n" if student_name else ""

        # Categorize risk level
        if predicted_score >= 75:
            risk = "low risk — on track"
        elif predicted_score >= 55:
            risk = "moderate risk — needs improvement"
        else:
            risk = "high risk — urgent intervention needed"

        # Build driver summary
        positive_drivers = [
            f"{feat} (impact: +{val:.1f})"
            for feat, val in shap_explanation.items()
            if val > 0
        ]
        negative_drivers = [
            f"{feat} (impact: {val:.1f})"
            for feat, val in shap_explanation.items()
            if val < 0
        ]

        prompt = f"""{name_line}
STUDENT PERFORMANCE ANALYSIS
=============================
Predicted Final Exam Score: {predicted_score:.1f} / 100
Risk Category: {risk}

RAW METRICS (current semester):
- Weekly study hours:        {student_data.get('hours_studied_per_week', 'N/A'):.1f} hrs
- Class attendance:          {student_data.get('attendance_percentage', 'N/A'):.1f}%
- Assignment completion:     {student_data.get('assignments_completion_rate', 'N/A'):.1f}%
- Previous exam score:       {student_data.get('previous_exam_score', 'N/A'):.1f} / 100
- Average sleep per night:   {student_data.get('sleep_hours_per_night', 'N/A'):.1f} hrs

WHAT IS HELPING THIS STUDENT:
{chr(10).join('- ' + d for d in positive_drivers) if positive_drivers else '- No significant positive factors identified'}

WHAT IS HURTING THIS STUDENT:
{chr(10).join('- ' + d for d in negative_drivers) if negative_drivers else '- No significant negative factors identified'}

Please write a personalized counseling report for this student based on the analysis above.
"""
        return prompt.strip()


# ------------------------------------------------------------------
# Prompt builder (standalone utility)
# ------------------------------------------------------------------

class PromptBuilder:
    """
    Builds structured prompts for various feedback scenarios.
    Separated from the generator for testability and reuse.
    """

    @staticmethod
    def at_risk_prompt(student_data: dict, score: float) -> str:
        """Specialized prompt for at-risk intervention."""
        return f"""
This student has a predicted score of {score:.1f}/100 and is at HIGH RISK of failing.
Study hours: {student_data.get('hours_studied_per_week', 0):.0f} hrs/week
Attendance: {student_data.get('attendance_percentage', 0):.0f}%

Write an urgent but supportive intervention message that:
1. Acknowledges their situation with empathy
2. Identifies the single most impactful change they can make
3. Suggests a concrete first step for this week
"""

    @staticmethod
    def high_performer_prompt(student_data: dict, score: float) -> str:
        """Specialized prompt for high-performing students."""
        return f"""
This student is predicted to score {score:.1f}/100 — excellent performance.
Write a brief encouraging message that:
1. Acknowledges their strong performance
2. Suggests how they could help peers or go deeper in the subject
"""


if __name__ == "__main__":
    # Example usage — requires ANTHROPIC_API_KEY in environment
    generator = StudentFeedbackGenerator()

    student = {
        "hours_studied_per_week":      12.0,
        "attendance_percentage":       65.0,
        "assignments_completion_rate": 70.0,
        "previous_exam_score":         55.0,
        "sleep_hours_per_night":        5.5,
    }

    shap_explanation = {
        "Weekly Study Hours":      -4.2,
        "Class Attendance":        -6.1,
        "Assignment Completion":   -2.3,
        "Previous Exam Score":     -8.0,
        "Sleep Hours":             -5.5,
    }

    feedback = generator.generate(
        student_data=student,
        predicted_score=52.3,
        shap_explanation=shap_explanation,
        student_name="Alex",
    )

    print("\n" + "="*60)
    print("GENERATED STUDENT FEEDBACK")
    print("="*60)
    print(feedback)
