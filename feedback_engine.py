import os
import json
import time
import math
import tempfile
from collections import defaultdict
from config import PROJECT_ROOT

FEEDBACK_DIR = os.path.join(PROJECT_ROOT, "feedback")
RATINGS_FILE = os.path.join(FEEDBACK_DIR, "ratings.json")
QA_PAIRS_FILE = os.path.join(FEEDBACK_DIR, "qa_pairs.json")

RATE_LIMIT_SECONDS = 60
RATE_LIMIT_PER_SESSION = 20

NEGATIVE_REASONS = {
    "inaccurate": "回答不准确",
    "irrelevant": "来源与问题无关",
    "incomplete": "回答不完整",
    "format_bad": "格式/排版混乱",
    "other": "其他原因",
}

WILSON_Z = 1.959964
DEDUP_THRESHOLD = 0.85


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


def _wilson_score(positive, total):
    if total == 0:
        return 0.0
    p = positive / total
    n = total
    z = WILSON_Z
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, min(1.0, (center - spread) / denom))


def _question_similarity(q1, q2):
    s1, s2 = set(q1), set(q2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


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


def record_rating(question, answer, score, params=None, reason=None):
    _ensure_dir()
    entry = {
        "question": question[:500],
        "answer": answer[:1000],
        "score": score,
        "timestamp": time.time(),
        "params": params or {},
    }
    if score < 0 and reason:
        entry["reason"] = reason
    ratings = _load_json(RATINGS_FILE)
    ratings.append(entry)
    _save_json(RATINGS_FILE, ratings)

    if score >= 1:
        qa_pairs = _load_json(QA_PAIRS_FILE)
        dup = False
        for existing in qa_pairs:
            if _question_similarity(question, existing.get("question", "")) > DEDUP_THRESHOLD:
                if score > existing.get("score", 0):
                    existing["question"] = question[:500]
                    existing["answer"] = answer[:1000]
                    existing["score"] = score
                    existing["timestamp"] = entry["timestamp"]
                    existing["params"] = params or {}
                    existing["source"] = "user_feedback"
                dup = True
                break
        if not dup:
            qa_pairs.append({
                "question": question[:500],
                "answer": answer[:1000],
                "source": "user_feedback",
                "score": score,
                "timestamp": entry["timestamp"],
                "params": params or {},
            })
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
            "health_score": 0,
            "by_k": {},
            "by_threshold": {},
            "by_search_type": {},
            "by_prompt_mode": {},
            "by_reason": {},
            "recent_trend": [],
            "timeline": [],
        }

    positive = sum(1 for r in ratings if r.get("score", 0) >= 1)
    negative = sum(1 for r in ratings if r.get("score", 0) < 0)
    total = len(ratings)

    by_k = defaultdict(lambda: {"total": 0, "positive": 0})
    by_threshold = defaultdict(lambda: {"total": 0, "positive": 0})
    by_search_type = defaultdict(lambda: {"total": 0, "positive": 0})
    by_prompt_mode = defaultdict(lambda: {"total": 0, "positive": 0})
    by_reason = defaultdict(int)

    for r in ratings:
        p = r.get("params", {})
        k_val = str(p.get("k", "unknown"))
        thresh_val = p.get("similarity_threshold")
        search_val = p.get("search_type", "unknown")
        prompt_val = p.get("prompt_mode", "unknown")
        s = r.get("score", 0)

        by_k[k_val]["total"] += 1
        if s >= 1:
            by_k[k_val]["positive"] += 1

        t_key = str(thresh_val) if thresh_val else "off"
        by_threshold[t_key]["total"] += 1
        if s >= 1:
            by_threshold[t_key]["positive"] += 1

        by_search_type[search_val]["total"] += 1
        if s >= 1:
            by_search_type[search_val]["positive"] += 1

        by_prompt_mode[prompt_val]["total"] += 1
        if s >= 1:
            by_prompt_mode[prompt_val]["positive"] += 1

        if s < 0 and r.get("reason"):
            by_reason[r["reason"]] += 1

    recent = sorted(ratings, key=lambda x: x.get("timestamp", 0), reverse=True)[:10]
    recent_trend = []
    for r in recent:
        recent_trend.append({
            "question": r.get("question", "")[:50],
            "score": r.get("score", 0),
            "time": time.strftime("%H:%M:%S", time.localtime(r.get("timestamp", 0))),
        })

    timeline_buckets = defaultdict(lambda: {"total": 0, "positive": 0})
    now = time.time()
    for r in ratings:
        ts = r.get("timestamp", 0)
        age_hours = max(0, (now - ts)) / 3600
        bucket = min(int(age_hours), 24)
        key = f"{bucket}h"
        timeline_buckets[key]["total"] += 1
        if r.get("score", 0) >= 1:
            timeline_buckets[key]["positive"] += 1

    timeline = []
    for i in range(25):
        key = f"{i}h"
        b = timeline_buckets.get(key, {"total": 0, "positive": 0})
        rate = round(b["positive"] / b["total"] * 100, 1) if b["total"] > 0 else None
        timeline.append({"bucket": key, "total": b["total"], "positive": b["positive"], "rate": rate})

    base_rate = positive / total if total > 0 else 0
    volume_score = min(100, total * 5)
    quality_score = base_rate * 100
    diversity_score = 0
    param_combos = len(by_k) + len(by_threshold) + len(by_search_type) + len(by_prompt_mode)
    if param_combos > 3:
        diversity_score = min(30, param_combos * 3)
    health_score = int(volume_score * 0.2 + quality_score * 0.6 + diversity_score * 0.2)

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_rate": round(base_rate * 100, 1),
        "health_score": min(100, max(0, health_score)),
        "by_k": dict(by_k),
        "by_threshold": dict(by_threshold),
        "by_search_type": dict(by_search_type),
        "by_prompt_mode": dict(by_prompt_mode),
        "by_reason": dict(by_reason),
        "recent_trend": recent_trend,
        "timeline": list(reversed(timeline)),
    }


def get_recommendation():
    stats = get_stats()
    recommendations = []

    k_data = stats["by_k"]
    if k_data:
        ranked_k = sorted(
            k_data.items(),
            key=lambda x: _wilson_score(x[1]["positive"], x[1]["total"]),
            reverse=True,
        )
        best = ranked_k[0]
        ws = _wilson_score(best[1]["positive"], best[1]["total"])
        if best[1]["total"] >= 1 and ws > 0.3:
            conf = "high" if best[1]["total"] >= 5 else "medium"
            recommendations.append({
                "param": "k",
                "value": int(best[0]) if best[0].isdigit() else best[0],
                "wilson_score": round(ws, 3),
                "confidence": conf,
                "raw_positive_rate": round(best[1]["positive"] / best[1]["total"], 2) if best[1]["total"] > 0 else 0,
                "sample_size": best[1]["total"],
                "reason": f"检索数量 k={best[0]} Wilson得分最高({ws:.0%})，基于{best[1]['total']}条反馈{'，置信度高' if conf=='high' else '，建议继续收集'}",
            })

    thresh_data = stats["by_threshold"]
    if thresh_data:
        ranked_t = sorted(
            thresh_data.items(),
            key=lambda x: _wilson_score(x[1]["positive"], x[1]["total"]),
            reverse=True,
        )
        best = ranked_t[0]
        ws = _wilson_score(best[1]["positive"], best[1]["total"])
        if best[1]["total"] >= 1 and ws > 0.3:
            conf = "high" if best[1]["total"] >= 5 else "medium"
            label = "关闭拒答阈值" if best[0] == "off" else f"阈值={best[0]}"
            recommendations.append({
                "param": "similarity_threshold",
                "value": None if best[0] == "off" else float(best[0]),
                "wilson_score": round(ws, 3),
                "confidence": conf,
                "raw_positive_rate": round(best[1]["positive"] / best[1]["total"], 2) if best[1]["total"] > 0 else 0,
                "sample_size": best[1]["total"],
                "reason": f"{label}时Wilson得分最高({ws:.0%})，基于{best[1]['total']}条反馈",
            })

    search_data = stats["by_search_type"]
    if search_data:
        ranked_s = sorted(
            search_data.items(),
            key=lambda x: _wilson_score(x[1]["positive"], x[1]["total"]),
            reverse=True,
        )
        best = ranked_s[0]
        ws = _wilson_score(best[1]["positive"], best[1]["total"])
        if best[1]["total"] >= 1 and ws > 0.3:
            conf = "high" if best[1]["total"] >= 5 else "medium"
            label_map = {"similarity": "相似度检索", "mmr": "MMR多样性检索"}
            recommendations.append({
                "param": "search_type",
                "value": best[0],
                "wilson_score": round(ws, 3),
                "confidence": conf,
                "raw_positive_rate": round(best[1]["positive"] / best[1]["total"], 2) if best[1]["total"] > 0 else 0,
                "sample_size": best[1]["total"],
                "reason": f"{label_map.get(best[0], best[0])}Wilson得分更高({ws:.0%})，基于{best[1]['total']}条反馈",
            })

    prompt_data = stats["by_prompt_mode"]
    if prompt_data:
        ranked_p = sorted(
            prompt_data.items(),
            key=lambda x: _wilson_score(x[1]["positive"], x[1]["total"]),
            reverse=True,
        )
        best = ranked_p[0]
        ws = _wilson_score(best[1]["positive"], best[1]["total"])
        if best[1]["total"] >= 1 and ws > 0.3:
            conf = "high" if best[1]["total"] >= 5 else "medium"
            recommendations.append({
                "param": "prompt_mode",
                "value": best[0],
                "wilson_score": round(ws, 3),
                "confidence": conf,
                "raw_positive_rate": round(best[1]["positive"] / best[1]["total"], 2) if best[1]["total"] > 0 else 0,
                "sample_size": best[1]["total"],
                "reason": f"提示词模式'{best[0]}'Wilson得分最高({ws:.0%})，基于{best[1]['total']}条反馈",
            })

    insights = []
    if len(search_data) >= 2:
        sim_w = _wilson_score(search_data.get("similarity", {}).get("positive", 0), search_data.get("similarity", {}).get("total", 1))
        mmr_w = _wilson_score(search_data.get("mmr", {}).get("positive", 0), search_data.get("mmr", {}).get("total", 1))
        if abs(sim_w - mmr_w) > 0.05:
            better = "MMR多样性" if mmr_w > sim_w else "相似度"
            diff = abs(sim_w - mmr_w)
            insights.append(f"检索策略对比：{better}检索比另一种Wilson得分高{diff:.0%}")
    by_r = stats.get("by_reason", {})
    if by_r:
        top_reason = max(by_r, key=by_r.get)
        count = by_r[top_reason]
        total_neg = sum(by_r.values())
        if total_neg > 0:
            pct = round(count / total_neg * 100)
            label = NEGATIVE_REASONS.get(top_reason, top_reason)
            insights.append(f"差评主因：'{label}'占{pct}%（{count}/{total_neg}条）")

    if not insights:
        if stats["total"] < 5:
            insights.append("反馈数据较少，继续使用后可获得更精准的分析洞察")
        else:
            insights.append("各参数组合表现稳定，系统运行健康")

    return {
        "recommendations": recommendations,
        "health_score": stats.get("health_score", 0),
        "insights": insights,
        "has_data": stats["total"] >= 3,
        "data_summary": f"共 {stats['total']} 条反馈 · 好评率 {stats['positive_rate']}%",
    }


def get_qa_pairs_count():
    qa_pairs = _load_json(QA_PAIRS_FILE)
    return len(qa_pairs)


def export_qa_pairs_to_docs():
    qa_pairs = _load_json(QA_PAIRS_FILE)
    if not qa_pairs:
        return 0

    from config import DATA_DIR
    export_path = os.path.join(DATA_DIR, "用户反馈问答对.md")

    lines = [
        "# MyLibrary RAG 用户反馈问答对（自进化知识库）\n",
        f"> 自动生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 问答对总数：{len(qa_pairs)} 条（已去重）\n",
        "---\n",
    ]
    for i, qa in enumerate(qa_pairs, 1):
        score_emoji = "👍" if qa.get("score", 0) >= 1 else "👎"
        ts_str = ""
        if qa.get("timestamp"):
            ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(qa["timestamp"]))
        params_str = ""
        if qa.get("params"):
            params_str = " · ".join(f"{k}={v}" for k, v in qa["params"].items() if v is not None)

        lines.append(f"## 问答对 {i} {score_emoji}\n")
        lines.append(f"**问题：** {qa.get('question', '')}\n\n")
        lines.append(f"**回答：**\n\n{qa.get('answer', '')}\n\n")
        lines.append(f"| 属性 | 值 |")
        lines.append(f"|------|---|")
        lines.append(f"| 评分 | {score_emoji} {'好评' if qa.get('score', 0) >= 1 else '差评'} ({qa.get('score', 0)}) |")
        lines.append(f"| 时间 | {ts_str} |")
        if params_str:
            lines.append(f"| 参数 | {params_str} |")
        lines.append(f"| 来源 | 用户反馈 |\n")
        lines.append("---\n")

    with open(export_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return len(qa_pairs)


def get_negative_reasons():
    return NEGATIVE_REASONS
