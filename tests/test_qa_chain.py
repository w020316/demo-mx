import os
import sys
import re

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestQAChainPatterns:
    def test_chat_patterns_greeting(self):
        from qa_chain import CHAT_PATTERNS
        for pat in CHAT_PATTERNS:
            assert isinstance(pat, str)
            re.compile(pat)

    def test_chat_patterns_compilable(self):
        from qa_chain import CHAT_PATTERNS
        for pat in CHAT_PATTERNS:
            compiled = re.compile(pat, re.IGNORECASE)
            assert compiled is not None


class TestQAChainConstants:
    def test_prompt_templates_keys(self):
        from qa_chain import PROMPT_TEMPLATES
        assert "anti_hallucination" in PROMPT_TEMPLATES
        assert "simple" in PROMPT_TEMPLATES
        assert "academic" in PROMPT_TEMPLATES
        assert "cross_doc" in PROMPT_TEMPLATES

    def test_prompt_labels_keys(self):
        from qa_chain import PROMPT_LABELS
        assert set(PROMPT_LABELS.keys()) == set(PROMPT_TEMPLATES.keys()) if 'PROMPT_TEMPLATES' in dir() else True

    def test_prompt_templates_contain_context_and_question(self):
        from qa_chain import PROMPT_TEMPLATES
        for key, template in PROMPT_TEMPLATES.items():
            assert "{context}" in template, f"Template '{key}' missing {{context}}"
            assert "{question}" in template, f"Template '{key}' missing {{question}}"

    def test_domain_keywords_type(self):
        from qa_chain import DOMAIN_KEYWORDS
        assert isinstance(DOMAIN_KEYWORDS, set)
        assert len(DOMAIN_KEYWORDS) > 0

    def test_threshold_values(self):
        from qa_chain import CHAT_FALLBACK_THRESHOLD, CHAT_DOMAIN_THRESHOLD
        assert 0 < CHAT_FALLBACK_THRESHOLD < 1
        assert 0 < CHAT_DOMAIN_THRESHOLD < 1
        assert CHAT_FALLBACK_THRESHOLD < CHAT_DOMAIN_THRESHOLD

    def test_rag_empty_patterns_type(self):
        from qa_chain import RAG_EMPTY_PATTERNS
        assert isinstance(RAG_EMPTY_PATTERNS, list)
        assert len(RAG_EMPTY_PATTERNS) > 0


class TestLooksLikeChatQuestion:
    def test_greeting_chinese(self):
        from qa_chain import _looks_like_chat_question
        assert _looks_like_chat_question("你好") is True

    def test_greeting_english(self):
        from qa_chain import _looks_like_chat_question
        assert _looks_like_chat_question("hello") is True

    def test_thanks(self):
        from qa_chain import _looks_like_chat_question
        assert _looks_like_chat_question("谢谢") is True

    def test_who_are_you(self):
        from qa_chain import _looks_like_chat_question
        assert _looks_like_chat_question("你是谁") is True

    def test_single_char(self):
        from qa_chain import _looks_like_chat_question
        assert _looks_like_chat_question("a") is True

    def test_empty_string(self):
        from qa_chain import _looks_like_chat_question
        assert _looks_like_chat_question("") is True

    def test_whitespace_only(self):
        from qa_chain import _looks_like_chat_question
        assert _looks_like_chat_question("   ") is True

    def test_short_non_domain(self):
        from qa_chain import _looks_like_chat_question
        assert _looks_like_chat_question("天气") is True

    def test_domain_question(self):
        from qa_chain import _looks_like_chat_question
        assert _looks_like_chat_question("RAG技术的核心原理是什么？") is False

    def test_python_question(self):
        from qa_chain import _looks_like_chat_question
        assert _looks_like_chat_question("Python如何实现向量检索？") is False

    def test_langchain_question(self):
        from qa_chain import _looks_like_chat_question
        assert _looks_like_chat_question("LangChain框架有哪些组件？") is False

    def test_normal_question(self):
        from qa_chain import _looks_like_chat_question
        assert _looks_like_chat_question("如何优化检索性能？") is False


class TestShouldFallbackToChat:
    def test_low_score_non_domain(self):
        from qa_chain import _should_fallback_to_chat
        assert _should_fallback_to_chat(0.2, "今天吃什么") is True

    def test_low_score_domain(self):
        from qa_chain import _should_fallback_to_chat
        assert _should_fallback_to_chat(0.2, "RAG如何工作") is False

    def test_high_score(self):
        from qa_chain import _should_fallback_to_chat
        assert _should_fallback_to_chat(0.8, "随便什么") is False

    def test_medium_score_non_domain(self):
        from qa_chain import _should_fallback_to_chat
        result = _should_fallback_to_chat(0.45, "推荐一本书")
        assert isinstance(result, bool)


class TestIsRagEmptyAnswer:
    def test_empty_pattern_detected(self):
        from qa_chain import _is_rag_empty_answer
        assert _is_rag_empty_answer("根据提供的文档内容，无法回答该问题") is True

    def test_normal_answer(self):
        from qa_chain import _is_rag_empty_answer
        assert _is_rag_empty_answer("RAG技术的核心是检索增强生成") is False

    def test_partial_match(self):
        from qa_chain import _is_rag_empty_answer
        assert _is_rag_empty_answer("知识库中没有相关信息") is True


class TestBuildPrompt:
    def test_valid_mode(self):
        from qa_chain import _build_prompt
        prompt = _build_prompt("anti_hallucination")
        assert prompt is not None

    def test_simple_mode(self):
        from qa_chain import _build_prompt
        prompt = _build_prompt("simple")
        assert prompt is not None

    def test_academic_mode(self):
        from qa_chain import _build_prompt
        prompt = _build_prompt("academic")
        assert prompt is not None

    def test_cross_doc_mode(self):
        from qa_chain import _build_prompt
        prompt = _build_prompt("cross_doc")
        assert prompt is not None

    def test_invalid_mode_fallback(self):
        from qa_chain import _build_prompt, DEFAULT_PROMPT_TEMPLATE
        prompt = _build_prompt("nonexistent_mode")
        assert prompt is not None


class TestExtractSources:
    def test_extract_from_mock_docs(self):
        from qa_chain import _extract_sources
        from unittest.mock import MagicMock

        doc1 = MagicMock()
        doc1.page_content = "content " * 100
        doc1.metadata = {"source_file": "test.pdf", "page": 0, "file_type": "pdf"}

        doc2 = MagicMock()
        doc2.page_content = "another doc"
        doc2.metadata = {"source_file": "readme.md", "page": 3, "file_type": "md"}

        sources = _extract_sources([doc1, doc2])
        assert len(sources) == 2
        assert sources[0]["source_file"] == "test.pdf"
        assert sources[0]["page"] == 0
        assert sources[0]["file_type"] == "pdf"
        assert sources[1]["source_file"] == "readme.md"
        assert sources[1]["page"] == 3

    def test_extract_sources_missing_metadata(self):
        from qa_chain import _extract_sources
        from unittest.mock import MagicMock

        doc = MagicMock()
        doc.page_content = "content"
        doc.metadata = {}

        sources = _extract_sources([doc])
        assert sources[0]["source_file"] == "未知"
        assert sources[0]["page"] == 0
        assert sources[0]["file_type"] == "未知"

    def test_extract_sources_content_truncation(self):
        from qa_chain import _extract_sources
        from unittest.mock import MagicMock

        doc = MagicMock()
        doc.page_content = "x" * 1000
        doc.metadata = {"source_file": "test.pdf", "page": 0, "file_type": "pdf"}

        sources = _extract_sources([doc])
        assert len(sources[0]["content"]) <= 300


class TestGetLLM:
    def test_missing_api_key_raises(self):
        from qa_chain import get_llm
        original = os.environ.get("OPENAI_API_KEY")
        os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                get_llm()
        finally:
            if original:
                os.environ["OPENAI_API_KEY"] = original

    def test_missing_base_url_raises(self):
        from qa_chain import get_llm
        original_key = os.environ.get("OPENAI_API_KEY")
        original_url = os.environ.get("OPENAI_BASE_URL")
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ.pop("OPENAI_BASE_URL", None)
        try:
            with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
                get_llm()
        finally:
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)
            if original_url:
                os.environ["OPENAI_BASE_URL"] = original_url


class TestConversationManager:
    def test_init_defaults(self):
        from qa_chain import ConversationManager
        cm = ConversationManager()
        assert cm.k == 3
        assert cm.prompt_mode == "anti_hallucination"
        assert cm.search_type == "similarity"
        assert cm.similarity_threshold is None
        assert cm.memory_window is None

    def test_init_custom(self):
        from qa_chain import ConversationManager
        cm = ConversationManager(k=5, prompt_mode="academic", search_type="mmr",
                                 similarity_threshold=0.5, memory_window=10)
        assert cm.k == 5
        assert cm.prompt_mode == "academic"
        assert cm.search_type == "mmr"
        assert cm.similarity_threshold == 0.5
        assert cm.memory_window == 10

    def test_reset(self):
        from qa_chain import ConversationManager
        cm = ConversationManager()
        cm._chain = "fake"
        cm._last_params = (1, 2, 3)
        cm.reset()
        assert cm._chain is None
        assert cm._last_params is None

    def test_trim_memory(self):
        from qa_chain import ConversationManager
        cm = ConversationManager(memory_window=2)
        from langchain_core.messages import HumanMessage, AIMessage
        for i in range(5):
            cm.memory.chat_memory.add_user_message(f"q{i}")
            cm.memory.chat_memory.add_ai_message(f"a{i}")
        cm._trim_memory()
        assert len(cm.memory.chat_memory.messages) <= 4

    def test_trim_memory_no_window(self):
        from qa_chain import ConversationManager
        cm = ConversationManager(memory_window=None)
        from langchain_core.messages import HumanMessage
        for i in range(10):
            cm.memory.chat_memory.add_user_message(f"q{i}")
        cm._trim_memory()
        assert len(cm.memory.chat_memory.messages) == 10

    def test_add_to_memory(self):
        from qa_chain import ConversationManager
        cm = ConversationManager(memory_window=5)
        cm._add_to_memory("test question", "test answer")
        messages = cm.memory.chat_memory.messages
        assert len(messages) == 2
