import os
import sys
import json
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestIntegrationFeedbackToRecommendation:
    @pytest.fixture(autouse=True)
    def setup_temp_dir(self, tmp_path):
        self.temp_dir = tmp_path
        self.feedback_dir = os.path.join(self.temp_dir, "feedback")
        os.makedirs(self.feedback_dir, exist_ok=True)
        self.ratings_file = os.path.join(self.feedback_dir, "ratings.json")
        self.qa_pairs_file = os.path.join(self.feedback_dir, "qa_pairs.json")
        import feedback_engine
        self.fe = feedback_engine
        self.original_ratings_file = feedback_engine.RATINGS_FILE
        self.original_qa_pairs_file = feedback_engine.QA_PAIRS_FILE
        self.original_feedback_dir = feedback_engine.FEEDBACK_DIR
        feedback_engine.RATINGS_FILE = self.ratings_file
        feedback_engine.QA_PAIRS_FILE = self.qa_pairs_file
        feedback_engine.FEEDBACK_DIR = self.feedback_dir
        yield
        feedback_engine.RATINGS_FILE = self.original_ratings_file
        feedback_engine.QA_PAIRS_FILE = self.original_qa_pairs_file
        feedback_engine.FEEDBACK_DIR = self.original_feedback_dir

    def test_full_feedback_loop(self):
        self.fe.record_rating("What is RAG?", "RAG is retrieval augmented generation", 1,
                              {"k": 3, "search_type": "similarity", "prompt_mode": "anti_hallucination",
                               "similarity_threshold": None})
        self.fe.record_rating("How does LangChain work?", "LangChain is a framework", 1,
                              {"k": 3, "search_type": "similarity", "prompt_mode": "anti_hallucination",
                               "similarity_threshold": None})
        self.fe.record_rating("What is Python?", "Python is a language", -1,
                              {"k": 5, "search_type": "mmr", "prompt_mode": "academic",
                               "similarity_threshold": 0.5}, reason="inaccurate")

        stats = self.fe.get_stats()
        assert stats["total"] == 3
        assert stats["positive"] == 2
        assert stats["negative"] == 1

        rec = self.fe.get_recommendation(stats)
        assert len(rec["recommendations"]) > 0
        assert rec["has_data"] is True

        qa_count = self.fe.get_qa_pairs_count()
        assert qa_count == 2

    def test_self_evolution_export_and_reingest(self, tmp_path):
        self.fe.record_rating("evolution q", "evolution a", 1, {"k": 3})
        import feedback_engine as fe_mod
        from config import DATA_DIR as CFG_DATA_DIR
        export_dir = os.path.join(self.temp_dir, "docs")
        os.makedirs(export_dir, exist_ok=True)
        fe_mod.DATA_DIR = export_dir
        try:
            count = fe_mod.export_qa_pairs_to_docs()
            assert count == 1
            export_file = os.path.join(export_dir, "用户反馈问答对.md")
            assert os.path.exists(export_file)
            with open(export_file, "r", encoding="utf-8") as f:
                content = f.read()
            assert "evolution q" in content
        finally:
            fe_mod.DATA_DIR = CFG_DATA_DIR


class TestIntegrationQAChainRouting:
    def test_chat_routing_greeting(self):
        from qa_chain import _looks_like_chat_question
        assert _looks_like_chat_question("你好") is True

    def test_chat_routing_domain_question(self):
        from qa_chain import _looks_like_chat_question
        assert _looks_like_chat_question("RAG技术如何工作？") is False

    def test_fallback_logic(self):
        from qa_chain import _should_fallback_to_chat
        assert _should_fallback_to_chat(0.1, "今天天气") is True
        assert _should_fallback_to_chat(0.1, "RAG原理") is False
        assert _should_fallback_to_chat(0.9, "任何问题") is False

    def test_empty_answer_detection(self):
        from qa_chain import _is_rag_empty_answer
        assert _is_rag_empty_answer("根据提供的文档内容，无法回答该问题") is True
        assert _is_rag_empty_answer("RAG是一种检索增强生成技术") is False


class TestIntegrationIngestPipeline:
    def test_load_and_split(self, tmp_path):
        from ingest import load_documents, split_documents
        f = tmp_path / "test.txt"
        f.write_text("这是第一段内容。\n\n这是第二段内容，包含更多的文字来确保分块能够正常工作。" * 10, encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) >= 1
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
        assert len(chunks) >= 1

    def test_hash_based_incremental_detection(self, tmp_path):
        from ingest import compute_file_hash, save_hashes, get_existing_hashes
        f = tmp_path / "doc.txt"
        f.write_text("content v1", encoding="utf-8")
        h1 = compute_file_hash(str(f))
        save_hashes(str(tmp_path), {"doc.txt": h1})

        loaded = get_existing_hashes(str(tmp_path))
        assert loaded["doc.txt"] == h1

        f.write_text("content v2", encoding="utf-8")
        h2 = compute_file_hash(str(f))
        assert h1 != h2


class TestIntegrationEndToEndFlow:
    def test_feedback_stats_recommendation_chain(self, tmp_path):
        feedback_dir = os.path.join(tmp_path, "feedback")
        os.makedirs(feedback_dir, exist_ok=True)
        ratings_file = os.path.join(feedback_dir, "ratings.json")
        qa_pairs_file = os.path.join(feedback_dir, "qa_pairs.json")

        import feedback_engine
        orig_r = feedback_engine.RATINGS_FILE
        orig_q = feedback_engine.QA_PAIRS_FILE
        orig_f = feedback_engine.FEEDBACK_DIR
        feedback_engine.RATINGS_FILE = ratings_file
        feedback_engine.QA_PAIRS_FILE = qa_pairs_file
        feedback_engine.FEEDBACK_DIR = feedback_dir

        try:
            questions = [
                "什么是RAG检索增强生成技术？",
                "LangChain框架的核心组件有哪些？",
                "Python中如何实现向量数据库？",
                "Streamlit的数据缓存机制是什么？",
                "如何优化文本分块策略？",
            ]
            for i, q in enumerate(questions):
                feedback_engine.record_rating(
                    q, f"answer_{i}", 1,
                    {"k": 3, "search_type": "similarity", "prompt_mode": "anti_hallucination"}
                )
            for i in range(3):
                feedback_engine.record_rating(
                    f"差评问题关于性能优化方案_{i}", f"bad_a_{i}", -1,
                    {"k": 5, "search_type": "mmr", "prompt_mode": "academic"},
                    reason="inaccurate"
                )

            stats = feedback_engine.get_stats()
            assert stats["total"] == 8
            assert stats["positive"] == 5
            assert stats["negative"] == 3

            rec = feedback_engine.get_recommendation(stats)
            assert len(rec["recommendations"]) > 0

            qa_count = feedback_engine.get_qa_pairs_count()
            assert qa_count == 5

            assert 0 <= stats["health_score"] <= 100
            assert len(stats["timeline"]) == 25
            assert len(stats["recent_trend"]) <= 10
        finally:
            feedback_engine.RATINGS_FILE = orig_r
            feedback_engine.QA_PAIRS_FILE = orig_q
            feedback_engine.FEEDBACK_DIR = orig_f
