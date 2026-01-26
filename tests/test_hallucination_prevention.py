"""
Integration tests for hallucination prevention mechanisms.
Tests constraint enforcement, prompt engineering, and document grounding.
"""

from src.prompt_builder import build_system_prompts


class TestHallucinationPreventionConstraints:
    """Test hallucination prevention through prompt constraints."""

    def test_system_prompts_contain_critical_constraint(self):
        """Test that system prompts include the CRITICAL constraint."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)
        prompt_upper = prompt_text.upper()

        assert "CRITICAL" in prompt_text
        assert "DO NOT" in prompt_upper
        assert "GENERAL KNOWLEDGE" in prompt_upper

    def test_system_prompts_contain_rejection_message(self):
        """Test that system prompts specify rejection for out-of-scope questions."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "not known to me" in prompt_text

    def test_system_prompts_contain_etymology_example(self):
        """Test that system prompts explicitly mention example topics."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Check for actual topics mentioned in the prompt config
        assert "evolution of languages" in prompt_text.lower()
        assert "ancient civilizations" in prompt_text.lower()

    def test_system_prompts_forbid_general_knowledge(self):
        """Test that system prompts explicitly forbid general knowledge usage."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "general knowledge" in prompt_text.lower()
        assert "fallback" in prompt_text.lower()

    def test_system_prompts_contain_no_inference_rule(self):
        """Test that system prompts forbid inference beyond explicit statements."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "infer" in prompt_text.lower() or "speculation" in prompt_text.lower()
        assert "explicitly" in prompt_text.lower()

    def test_system_prompts_contain_document_grounding_requirement(self):
        """Test that system prompts require grounding in documents."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "document" in prompt_text.lower()
        assert "ground" in prompt_text.lower() or "grounded" in prompt_text.lower()
