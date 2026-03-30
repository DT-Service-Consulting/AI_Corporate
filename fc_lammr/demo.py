"""Minimal working demo for FC-LAMMR."""

from __future__ import annotations

import json
from types import SimpleNamespace

from fc_lammr.fc_lammr_router import FCLAMMRRouter


class DemoLLMClient:
    """Deterministic demo client that triggers ToM and rerouting behavior."""

    def create_chat_completion(self, *, messages, model=None, temperature=0.2, max_tokens=800):
        user_text = messages[-1]["content"]
        if "Return only valid JSON" in user_text:
            payload = {
                "surface_task": "Locate section 4.2",
                "underlying_goal": "Assess whether subcontractor qualification requirements create tender compliance risk.",
                "inferred_phase": "risk_assessment",
                "task_type": "tender_evaluation",
                "confidence": 0.86,
                "reasoning": "The query sounds extractive but the tender context implies a procurement suitability check.",
            }
            content = json.dumps(payload)
        elif "HANDOFF CONTEXT:" in user_text:
            content = "YES. Section 4.2 should be assessed as a tender compliance issue because subcontractor qualifications affect eligibility and challenge risk."
        else:
            content = (
                "It is unclear whether Section 9.9 applies. This may depend on external procurement policy. "
                "The NOVATION term is referenced, but it does not appear in the document."
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(total_tokens=120),
        )


def main() -> None:
    """Run one complete FC-LAMMR routing cycle and print the explanation."""
    router = FCLAMMRRouter(
        llm_client=DemoLLMClient(),
        pattern_library_path="fc_lammr/pattern_library.json",
    )
    query = "What does section 4.2 say about subcontractor qualifications?"
    document = (
        "Tender Conditions. Section 4.2 requires subcontractors to hold three years of comparable experience "
        "and maintain current safety accreditation."
    )
    state = router.route(query, document)
    print("Final output:")
    print(state.final_output)
    print("\nRoute explanation:")
    print(router.explain_route(state))


if __name__ == "__main__":
    main()
