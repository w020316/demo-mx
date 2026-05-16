import os
import sys
import json
import time
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestFeedbackEngineUnit:
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

    def test_wilson_score_zero_total(self):
        assert self.fe._wilson_score(0, 0) == 0.0

    def test_wilson_score_all_positive(self):
        score = self.fe._wilson_score(10, 10)
        assert 0.0 < score <= 1.0

    def test_wilson_score_all_negative(self):
        score = self.fe._wilson_score(0, 10)
        assert 0.0 <= score < 0.5

    def test_wilson_score_mixed(self):
        score = self.fe._wilson_score(5, 10)
        assert 0.0 < score < 1.0

    def test_wilson_score_type_hints(self):
        result = self.fe._wilson_score(1, 5)
        assert isinstance(result, float)

    def test_sanitize_string_normal(self):
        assert self.fe._sanitize_string("hello", 10) == "hello"

    def test_sanitize_string_truncation(self):
        long_str = "a" * 100
        assert len(self.fe._sanitize_string(long_str, 10)) == 10

    def test_sanitize_string_non_str(self):
        assert self.fe._sanitize_string(12345, 10) == "12345"

    def test_check_abuse_session_limit(self):
        allowed, msg = self.fe.check_abuse("test", 20)
        assert not allowed
        assert "上限" in msg

    def test_check_abuse_session_under_limit(self):
        allowed, msg = self.fe.check_abuse("test", 5)
        assert allowed

    def test_check_abuse_empty_question(self):
        allowed, msg = self.fe.check_abuse("", 0)
        assert not allowed

    def test_check_abuse_non_string_question(self):
        allowed, msg = self.fe.check_abuse(123, 0)
        assert not allowed

    def test_check_abuse_rate_limiting(self):
        self.fe.record_rating("rate_test", "answer", 1)
        allowed, msg = self.fe.check_abuse("rate_test", 0)
        assert not allowed
        assert "秒" in msg

    def test_check_abuse_rate_limit_expired(self):
        entry = {
            "question": "old_test",
            "answer": "a",
            "score": 1,
            "timestamp": time.time() - 120,
            "params": {},
        }
        with open(self.ratings_file, "w", encoding="utf-8") as f:
            json.dump([entry], f)
        allowed, msg = self.fe.check_abuse("old_test", 0)
        assert allowed

    def test_record_rating_positive(self):
        entry = self.fe.record_rating("q1", "a1", 1, {"k": 3})
        assert entry["question"] == "q1"
        assert entry["score"] == 1
        assert entry["params"] == {"k": 3}

    def test_record_rating_negative_with_reason(self):
        entry = self.fe.record_rating("q2", "a2", -1, reason="inaccurate")
        assert entry["score"] == -1
        assert entry["reason"] == "inaccurate"

    def test_record_rating_negative_without_reason(self):
        entry = self.fe.record_rating("q3", "a3", -1)
        assert "reason" not in entry

    def test_record_rating_invalid_score(self):
        entry = self.fe.record_rating("q4", "a4", "bad")
        assert entry["score"] == 0

    def test_record_rating_truncation(self):
        long_q = "x" * 1000
        long_a = "y" * 2000
        entry = self.fe.record_rating(long_q, long_a, 1)
        assert len(entry["question"]) <= 500
        assert len(entry["answer"]) <= 1000

    def test_record_rating_creates_qa_pair(self):
        self.fe.record_rating("q_good", "good answer", 1)
        count = self.fe.get_qa_pairs_count()
        assert count == 1

    def test_record_rating_negative_no_qa_pair(self):
        self.fe.record_rating("q_bad", "bad answer", -1)
        count = self.fe.get_qa_pairs_count()
        assert count == 0

    def test_record_rating_dedup(self):
        self.fe.record_rating("same question", "answer1", 1)
        self.fe.record_rating("same question", "answer2", 1)
        count = self.fe.get_qa_pairs_count()
        assert count == 1

    def test_record_rating_dedup_higher_score_replaces(self):
        self.fe.record_rating("dup q", "low answer", 1)
        self.fe.record_rating("dup q", "high answer", 2)
        qa_pairs = self.fe._load_json(self.qa_pairs_file)
        assert qa_pairs[0]["answer"] == "high answer"

    def test_record_rating_max_entries(self):
        for i in range(15):
            self.fe.record_rating(f"q_{i}", f"a_{i}", 1)
        self.fe.MAX_RATINGS_ENTRIES = 10
        self.fe.record_rating("overflow_q", "overflow_a", 1)
        ratings = self.fe._load_json(self.ratings_file)
        assert len(ratings) <= 11
        self.fe.MAX_RATINGS_ENTRIES = 10000

    def test_get_stats_empty(self):
        stats = self.fe.get_stats()
        assert stats["total"] == 0
        assert stats["positive"] == 0
        assert stats["negative"] == 0
        assert stats["positive_rate"] == 0.0
        assert stats["health_score"] == 0

    def test_get_stats_with_data(self):
        self.fe.record_rating("q1", "a1", 1, {"k": 3, "search_type": "similarity", "prompt_mode": "simple"})
        self.fe.record_rating("q2", "a2", -1, {"k": 5, "search_type": "mmr", "prompt_mode": "academic"}, reason="inaccurate")
        self.fe.record_rating("q3", "a3", 1, {"k": 3, "search_type": "similarity", "prompt_mode": "simple"})
        stats = self.fe.get_stats()
        assert stats["total"] == 3
        assert stats["positive"] == 2
        assert stats["negative"] == 1
        assert stats["positive_rate"] == pytest.approx(66.7, abs=0.1)
        assert "3" in stats["by_k"]
        assert "5" in stats["by_k"]
        assert "similarity" in stats["by_search_type"]
        assert "inaccurate" in stats["by_reason"]

    def test_get_stats_health_score_range(self):
        for i in range(10):
            self.fe.record_rating(f"q_{i}", f"a_{i}", 1, {"k": 3})
        stats = self.fe.get_stats()
        assert 0 <= stats["health_score"] <= 100

    def test_get_stats_timeline(self):
        self.fe.record_rating("q1", "a1", 1)
        stats = self.fe.get_stats()
        assert isinstance(stats["timeline"], list)
        assert len(stats["timeline"]) == 25

    def test_get_stats_recent_trend(self):
        self.fe.record_rating("q1", "a1", 1)
        stats = self.fe.get_stats()
        assert isinstance(stats["recent_trend"], list)
        assert len(stats["recent_trend"]) == 1

    def test_get_recommendation_empty(self):
        rec = self.fe.get_recommendation({"total": 0})
        assert rec["recommendations"] == []
        assert len(rec["insights"]) > 0

    def test_get_recommendation_with_data(self):
        stats = {
            "total": 10,
            "positive_rate": 80.0,
            "health_score": 75,
            "by_k": {"3": {"total": 8, "positive": 7}, "5": {"total": 2, "positive": 1}},
            "by_threshold": {"off": {"total": 10, "positive": 8}},
            "by_search_type": {"similarity": {"total": 6, "positive": 5}, "mmr": {"total": 4, "positive": 3}},
            "by_prompt_mode": {"anti_hallucination": {"total": 10, "positive": 8}},
            "by_reason": {},
        }
        rec = self.fe.get_recommendation(stats)
        assert len(rec["recommendations"]) > 0
        assert rec["has_data"] is True

    def test_get_recommendation_insights_search_comparison(self):
        stats = {
            "total": 10,
            "positive_rate": 50.0,
            "health_score": 50,
            "by_k": {},
            "by_threshold": {},
            "by_search_type": {
                "similarity": {"total": 10, "positive": 9},
                "mmr": {"total": 10, "positive": 1},
            },
            "by_prompt_mode": {},
            "by_reason": {},
        }
        rec = self.fe.get_recommendation(stats)
        has_search_insight = any("检索策略对比" in i for i in rec["insights"])
        assert has_search_insight

    def test_get_recommendation_insights_negative_reason(self):
        stats = {
            "total": 5,
            "positive_rate": 20.0,
            "health_score": 30,
            "by_k": {},
            "by_threshold": {},
            "by_search_type": {},
            "by_prompt_mode": {},
            "by_reason": {"inaccurate": 3, "irrelevant": 1},
        }
        rec = self.fe.get_recommendation(stats)
        has_reason_insight = any("差评主因" in i for i in rec["insights"])
        assert has_reason_insight

    def test_get_qa_pairs_count_empty(self):
        assert self.fe.get_qa_pairs_count() == 0

    def test_get_qa_pairs_count_with_data(self):
        self.fe.record_rating("q1", "a1", 1)
        self.fe.record_rating("q2", "a2", 1)
        assert self.fe.get_qa_pairs_count() == 2

    def test_export_qa_pairs_empty(self):
        count = self.fe.export_qa_pairs_to_docs()
        assert count == 0

    def test_export_qa_pairs_with_data(self, tmp_path):
        self.fe.record_rating("export_q", "export_a", 1)
        import feedback_engine
        original_data_dir = feedback_engine.DATA_DIR if hasattr(feedback_engine, 'DATA_DIR') else None
        from config import DATA_DIR as CFG_DATA_DIR
        export_dir = os.path.join(self.temp_dir, "docs")
        os.makedirs(export_dir, exist_ok=True)
        import feedback_engine as fe_mod
        fe_mod.DATA_DIR = export_dir
        try:
            count = fe_mod.export_qa_pairs_to_docs()
            assert count == 1
            export_file = os.path.join(export_dir, "用户反馈问答对.md")
            assert os.path.exists(export_file)
            with open(export_file, "r", encoding="utf-8") as f:
                content = f.read()
            assert "export_q" in content
            assert "export_a" in content
        finally:
            fe_mod.DATA_DIR = CFG_DATA_DIR

    def test_get_negative_reasons(self):
        reasons = self.fe.get_negative_reasons()
        assert isinstance(reasons, dict)
        assert "inaccurate" in reasons
        assert "irrelevant" in reasons
        assert "incomplete" in reasons
        assert "format_bad" in reasons
        assert "other" in reasons

    def test_load_json_nonexistent(self):
        result = self.fe._load_json("/nonexistent/path.json")
        assert result == []

    def test_load_json_invalid_json(self):
        bad_file = os.path.join(self.feedback_dir, "bad.json")
        with open(bad_file, "w") as f:
            f.write("not valid json{{{")
        result = self.fe._load_json(bad_file)
        assert result == []

    def test_load_json_not_list(self):
        not_list_file = os.path.join(self.feedback_dir, "notlist.json")
        with open(not_list_file, "w") as f:
            json.dump({"key": "value"}, f)
        result = self.fe._load_json(not_list_file)
        assert result == []

    def test_save_and_load_json_roundtrip(self):
        data = [{"question": "test", "answer": "ans", "score": 1}]
        test_file = os.path.join(self.feedback_dir, "roundtrip.json")
        self.fe._save_json(test_file, data)
        result = self.fe._load_json(test_file)
        assert result == data

    def test_question_similarity_identical(self):
        assert self.fe._question_similarity("hello", "hello") == 1.0

    def test_question_similarity_different(self):
        sim = self.fe._question_similarity("hello", "world")
        assert 0.0 <= sim < 1.0

    def test_question_similarity_empty(self):
        sim = self.fe._question_similarity("", "")
        assert sim == 1.0

    def test_rank_by_wilson(self):
        data = {
            "a": {"positive": 9, "total": 10},
            "b": {"positive": 1, "total": 10},
            "c": {"positive": 5, "total": 10},
        }
        ranked = self.fe._rank_by_wilson(data)
        assert ranked[0][0] == "a"
        assert ranked[-1][0] == "b"

    def test_params_validation_in_stats(self):
        entry = {
            "question": "bad_params",
            "answer": "a",
            "score": 1,
            "timestamp": time.time(),
            "params": "not_a_dict",
        }
        with open(self.ratings_file, "w", encoding="utf-8") as f:
            json.dump([entry], f)
        stats = self.fe.get_stats()
        assert stats["total"] == 1
