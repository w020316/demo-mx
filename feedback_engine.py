import os
import json
import time
import tempfile
from config import PROJECT_ROOT

FEEDBACK_DIR = os.path.join(PROJECT_ROOT, "feedback")
RATINGS_FILE = os.path.join(FEEDBACK_DIR, "ratings.json")
QA_PAIRS_FILE = os.path.join(FEEDBACK_DIR, "qa_pairs.json")

RATE_LIMIT_SECONDS = 60
RATE_LIMIT_PER_SESSION = 20


def _ensure_dir():
    os.makedirs(FEEDBACK_DIR, exist_ok=True)


def _load_json(filepath):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_json(filepath, data):
    _ensure_dir()
    fd, tmp_path = tempfile.mkstemp(dir=FEEDBACK_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def check_abuse(question, session_ratings_count):
    if session_ratings_count >= RATE_LIMIT_PER_SESSION:
        return False, f"单次会话最多评分 {RATE_LIMIT_PER_SESSION} 次，已达上限"
    ratings = _load_json(RATINGS_FILE)
    now = time.time()
    for r in reversed(ratings):
        if r.get("question") == question:
            elapsed = now - r.get("timestamp", 0)
            if elapsed < RATE_LIMIT_SECONDS:
                remaining = int(RATE_LIMIT_SECONDS - elapsed)
                return False, f"同一问题 {RATE_LIMIT_SECONDS} 秒内不可重复评分，请等待 {remaining} 秒"
            break
    return True, ""


def record_rating(question, answer, score, params=None):
    _ensure_dir()
    entry = {
        "question": question[:500],
        "answer": answer[:1000],
        "score": score,
        "timestamp": time.time(),
        "params": params or {},
    }
    ratings = _load_json(RATINGS_FILE)
    ratings.append(entry)
    _save_json(RATINGS_FILE, ratings)

    if score >= 1:
        qa_pairs = _load_json(QA_PAIRS_FILE)
        qa_entry = {
            "question": question[:500],
            "answer": answer[:1000],
            "source": "user_feedback",
            "score": score,
            "timestamp": entry["timestamp"],
            "params": params or {},
        }
        qa_pairs.append(qa_entry)
        _save_json(QA_PAIRS_FILE, qa_pairs)

    return entry


def get_stats():
    ratings = _load_json(RATINGS_FILE)
    if not ratings:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "positive_rate": 0.0,
            "by_k": {},
            "by_threshold": {},
            "by_search_type": {},
            "by_prompt_mode": {},
            "recent_trend": [],
        }

    positive = sum(1 for r in ratings if r.get("score", 0) >= 1)
    negative = sum(1 for r in ratings if r.get("score", 0) < 1)
    total = len(ratings)

    by_k = {}
    by_threshold = {}
    by_search_type = {}
    by_prompt_mode = {}

    for r in ratings:
        p = r.get("params", {})
        k_val = p.get("k", "unknown")
        thresh_val = p.get("similarity_threshold", "off")
        search_val = p.get("search_type", "unknown")
        prompt_val = p.get("prompt_mode", "unknown")
        s = r.get("score", 0)

        if k_val not in by_k:
            by_k[k_val] = {"total": 0, "positive": 0}
        by_k[k_val]["total"] += 1
        if s >= 1:
            by_k[k_val]["positive"] += 1

        t_key = str(thresh_val) if thresh_val else "off"
        if t_key not in by_threshold:
            by_threshold[t_key] = {"total": 0, "positive": 0}
        by_threshold[t_key]["total"] += 1
        if s >= 1:
            by_threshold[t_key]["positive"] += 1

        if search_val not in by_search_type:
            by_search_type[search_val] = {"total": 0, "positive": 0}
        by_search_type[search_val]["total"] += 1
        if s >= 1:
            by_search_type[search_val]["positive"] += 1

        if prompt_val not in by_prompt_mode:
            by_prompt_mode[prompt_val] = {"total": 0, "positive": 0}
        by_prompt_mode[prompt_val]["total"] += 1
        if s >= 1:
            by_prompt_mode[prompt_val]["positive"] += 1

    recent = sorted(ratings, key=lambda x: x.get("timestamp", 0), reverse=True)[:10]
    recent_trend = []
    for r in recent:
        recent_trend.append({
            "question": r.get("question", "")[:50],
            "score": r.get("score", 0),
            "time": time.strftime("%H:%M:%S", time.localtime(r.get("timestamp", 0))),
        })

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_rate": round(positive / total * 100, 1) if total > 0 else 0.0,
        "by_k": by_k,
        "by_threshold": by_threshold,
        "by_search_type": by_search_type,
        "by_prompt_mode": by_prompt_mode,
        "recent_trend": recent_trend,
    }


def get_recommendation():
    stats = get_stats()
    if stats["total"] < 3:
        return "📊 反馈数据不足（需≥3条），暂无调整建议。继续使用系统后即可获得个性化推荐。"

    recommendations = []

    best_k = None
    best_k_rate = -1
    for k_val, data in stats["by_k"].items():
        if data["total"] >= 2:
            rate = data["positive"] / data["total"]
            if rate > best_k_rate:
                best_k_rate = rate
                best_k = k_val
    if best_k is not None and best_k_rate > 0.5:
        recommendations.append(f"🔍 检索数量 k={best_k} 好评率最高({best_k_rate:.0%})，建议优先使用")

    best_thresh = None
    best_thresh_rate = -1
    for t_val, data in stats["by_threshold"].items():
        if data["total"] >= 2:
            rate = data["positive"] / data["total"]
            if rate > best_thresh_rate:
                best_thresh_rate = rate
                best_thresh = t_val
    if best_thresh is not None:
        if best_thresh == "off":
            recommendations.append("🔓 关闭拒答阈值时好评率更高，当前文档库无需严格过滤")
        else:
            recommendations.append(f"🎯 拒答阈值={best_thresh} 好评率最高({best_thresh_rate:.0%})，建议采用")

    best_search = None
    best_search_rate = -1
    for s_val, data in stats["by_search_type"].items():
        if data["total"] >= 2:
            rate = data["positive"] / data["total"]
            if rate > best_search_rate:
                best_search_rate = rate
                best_search = s_val
    if best_search is not None:
        label = "相似度检索" if best_search == "similarity" else "MMR多样性检索"
        recommendations.append(f"🔎 {label}好评率更高({best_search_rate:.0%})，推荐使用")

    best_prompt = None
    best_prompt_rate = -1
    for p_val, data in stats["by_prompt_mode"].items():
        if data["total"] >= 2:
            rate = data["positive"] / data["total"]
            if rate > best_prompt_rate:
                best_prompt_rate = rate
                best_prompt = p_val
    if best_prompt is not None:
        recommendations.append(f"💬 提示词模式 '{best_prompt}' 好评率最高({best_prompt_rate:.0%})")

    if not recommendations:
        recommendations.append("📈 各参数组合表现接近，暂无明确优化方向，继续收集反馈中...")

    return "\n\n".join(recommendations)


def get_qa_pairs_count():
    qa_pairs = _load_json(QA_PAIRS_FILE)
    return len(qa_pairs)


def export_qa_pairs_to_docs():
    qa_pairs = _load_json(QA_PAIRS_FILE)
    if not qa_pairs:
        return 0

    from config import DATA_DIR
    export_path = os.path.join(DATA_DIR, "用户反馈问答对.txt")

    lines = ["=== MyLibrary RAG 用户反馈问答对（自进化知识库） ===\n"]
    for i, qa in enumerate(qa_pairs, 1):
        lines.append(f"\n--- 问答对 {i} ---")
        lines.append(f"问题: {qa.get('question', '')}")
        lines.append(f"回答: {qa.get('answer', '')}")
        lines.append(f"评分: {'👍好评' if qa.get('score', 0) >= 1 else '👎差评'}")
        ts = qa.get("timestamp", 0)
        if ts:
            lines.append(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}")
        lines.append("")

    with open(export_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return len(qa_pairs)
