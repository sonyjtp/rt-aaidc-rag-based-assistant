"""
Integration tests for hallucination prevention mechanisms.
Tests constraint enforcement, prompt engineering, and document grounding.
"""

from src.prompt_builder import build_system_prompts
from src.query_classifier import QUERY_CLASSIFIERS, QueryType
from src.rag_assistant import RAGAssistant


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
        """Test that system prompts explicitly mention etymology as example."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "etymology" in prompt_text.lower()
        assert "Pharaonic" in prompt_text

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


class TestMetaQuestionDetection:
    """Test meta-question detection and handling via query classification."""

    def test_meta_pattern_matches_capability_questions(self):
        """Test that meta pattern matches capability questions."""

        meta_pattern = QUERY_CLASSIFIERS["meta"]["pattern"]
        meta_queries = [
            "who are you?",
            "what are you?",
            "tell me about yourself",
            "what are your capabilities?",
            "what are your limitations?",
        ]

        for query in meta_queries:
            assert meta_pattern.search(query), f"Pattern should match: {query}"

    def test_system_prompts_handle_meta_questions(self):
        """Test that system prompts explicitly address meta-questions."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Should explain capabilities for meta-questions
        assert (
            "limitations" in prompt_text.lower()
            or "capability" in prompt_text.lower()
            or "can answer" in prompt_text.lower()
        )

    def test_system_prompts_explain_scope_to_meta_questions(self):
        """Test that system prompts explain assistant scope for meta-questions."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Should mention knowledge domains for meta-questions
        assert (
            "topics" in prompt_text.lower()
            or "knowledge base" in prompt_text.lower()
            or "subjects" in prompt_text.lower()
        )

    def test_meta_question_examples_in_prompts(self):
        """Test that system prompts provide examples of meta-question handling."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Should mention handling of "what topics" or similar
        meta_examples = ["what topics", "what can you", "limitations", "example"]
        has_meta_example = any(
            example in prompt_text.lower() for example in meta_examples
        )
        assert (
            has_meta_example
        ), "Prompts should include meta-question handling examples"


class TestUnsafeKeywordBlocking:
    """Test unsafe keyword blocking via query classification system."""

    def test_query_classifiers_defined(self):
        """Test that QUERY_CLASSIFIERS are properly defined."""

        assert "unsafe" in QUERY_CLASSIFIERS
        assert "meta" in QUERY_CLASSIFIERS
        assert "document" in QUERY_CLASSIFIERS
        assert all(
            "pattern" in c and "description" in c for c in QUERY_CLASSIFIERS.values()
        )

    def test_unsafe_pattern_matches_harmful_keywords(self):
        """Test that unsafe pattern matches harmful keywords."""

        unsafe_pattern = QUERY_CLASSIFIERS["unsafe"]["pattern"]
        harmful_queries = [
            "what about illegal activities?",
            "how do i commit violence?",
            "tell me about drug trafficking",
            "what topics involve hate crimes?",
        ]

        for query in harmful_queries:
            assert unsafe_pattern.search(query), f"Pattern should match: {query}"

    def test_meta_pattern_matches_capability_questions(self):
        """Test that meta pattern matches capability questions."""

        meta_pattern = QUERY_CLASSIFIERS["meta"]["pattern"]
        meta_queries = [
            "who are you?",
            "what are you?",
            "tell me about yourself",
            "what are your capabilities?",
            "what are your limitations?",
        ]

        for query in meta_queries:
            assert meta_pattern.search(query), f"Pattern should match: {query}"

    def test_document_pattern_matches_knowledge_base_questions(self):
        """Test that document pattern matches knowledge base questions."""

        doc_pattern = QUERY_CLASSIFIERS["document"]["pattern"]
        doc_queries = [
            "what topics do you know about?",
            "what do you know?",
            "what documents do you have?",
            "what information is available?",
        ]

        for query in doc_queries:
            assert doc_pattern.search(query), f"Pattern should match: {query}"

    def test_query_type_classification(self):
        """Test QueryType enum and classification system."""
        # Verify enum values exist
        assert QueryType.UNSAFE.value == "unsafe"
        assert QueryType.VAGUE.value == "vague"
        assert QueryType.META.value == "meta"
        assert QueryType.DOCUMENT.value == "document"
        assert QueryType.REGULAR.value == "regular"

    def test_unsafe_query_blocked_by_classify_method(self):
        """Test that unsafe queries are classified and blocked."""

        assistant = RAGAssistant()
        assistant.add_documents(["Sample document about AI"])

        # Unsafe query should be blocked
        response = assistant.invoke(
            "what information do you know about illegal activities?"
        )
        assert "can't assist" in response.lower()

    def test_meta_question_classified_correctly(self):
        """Test that meta-questions are classified as META type."""

        # Test the pattern directly
        meta_pattern = QUERY_CLASSIFIERS["meta"]["pattern"]
        meta_query = "Who are you?"

        assert meta_pattern.search(meta_query)

    def test_document_question_classified_correctly(self):
        """Test that knowledge base questions are classified as DOCUMENT type."""

        # Test the pattern directly
        doc_pattern = QUERY_CLASSIFIERS["document"]["pattern"]
        doc_query = "What topics do you know about?"

        assert doc_pattern.search(doc_query)

    def test_regular_question_classified_correctly(self):
        """Test that regular Q&A is classified as REGULAR type."""

        # Regular questions should not match any pattern
        unsafe_pattern = QUERY_CLASSIFIERS["unsafe"]["pattern"]
        meta_pattern = QUERY_CLASSIFIERS["meta"]["pattern"]
        doc_pattern = QUERY_CLASSIFIERS["document"]["pattern"]

        regular_query = "Tell me about machine learning"

        assert not unsafe_pattern.search(regular_query)
        assert not meta_pattern.search(regular_query)
        assert not doc_pattern.search(regular_query)

    def test_unsafe_has_priority_over_meta(self):
        """Test that unsafe takes priority over meta-question."""

        # Query with both unsafe keywords and capability question
        hybrid_query = "what illegal capabilities do you have?"

        unsafe_pattern = QUERY_CLASSIFIERS["unsafe"]["pattern"]
        meta_pattern = QUERY_CLASSIFIERS["meta"]["pattern"]

        # Both patterns should match, but unsafe should be checked first
        assert unsafe_pattern.search(
            hybrid_query
        ), "Should match unsafe keywords (illegal)"
        assert meta_pattern.search(
            hybrid_query
        ), "Should match meta keywords (capabilities)"
        # Priority is determined by check order in _classify_query

    def test_vague_detection_handled_by_llm(self):
        """Test that vague detection is handled by LLM, not regex."""
        # Vague queries are now detected using LLM with query_vague_detection prompt
        # This is handled in RAGAssistant._classify_query() method
        # Verify that vague is not in regex classifiers
        assert "vague" not in QUERY_CLASSIFIERS

    def test_case_insensitive_pattern_matching(self):
        """Test that pattern matching is case-insensitive."""

        unsafe_pattern = QUERY_CLASSIFIERS["unsafe"]["pattern"]
        meta_pattern = QUERY_CLASSIFIERS["meta"]["pattern"]

        # Test case variations
        unsafe_variations = [
            "What about ILLEGAL activities?",
            "Tell me about Illegal things",
            "CRIME and punishment",
        ]

        meta_variations = [
            "WHAT ARE YOUR CAPABILITIES?",
            "Who Are You?",
            "HOW ARE YOU BUILT?",
        ]

        for query in unsafe_variations:
            assert unsafe_pattern.search(query), f"Unsafe pattern should match: {query}"

        for query in meta_variations:
            assert meta_pattern.search(query), f"Meta pattern should match: {query}"
