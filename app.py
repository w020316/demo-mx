import streamlit as st
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylibs"))

st.set_page_config(
    page_title="MyLibrary RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp { background: #fafbfc; }
.source-badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:600; margin-right:3px; }
.badge-pdf { background:#fef2f2; color:#dc2626; }
.badge-txt { background:#eff6ff; color:#2563eb; }
.badge-md { background:#f0fdf4; color:#16a34a; }
.welcome-box { text-align:center; padding:2rem 1.2rem; background:linear-gradient(135deg,#f8fafc 0%,#e2e8f0 100%); border-radius:14px; margin:1rem 0; border:1px solid #e2e8f0; }
.welcome-box h2 { font-size:1.35rem; color:#1e293b; margin-bottom:.4rem; }
.welcome-box p { color:#64748b; font-size:.9rem; line-height:1.65; }
.fb-btn:hover { transform:scale(1.04); transition:transform .15s; }
.mode-badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:600; margin-left:6px; vertical-align:middle; }
.mode-rag { background:#dbeafe; color:#1d4ed8; }
.mode-chat { background:#fef3c7; color:#b45309; }
.mode-reject { background:#fee2e2; color:#dc2626; }

.health-gauge { text-align:center; padding:12px; border-radius:12px; margin:8px 0; }
.health-excellent { background:linear-gradient(135deg,#dcfce7,#bbf7d0); border:1.5px solid #22c55e; }
.health-good { background:linear-gradient(135deg,#fef9c3,#fef08a); border:1.5px solid #eab308; }
.health-fair { background:linear-gradient(135deg,#ffedd5,#fed7aa); border:1.5px solid #f97316; }
.health-poor { background:linear-gradient(135deg,#fee2e2,#fecaca); border:1.5px solid #ef4444; }
.health-score { font-size:2rem; font-weight:800; line-height:1; }
.health-label { font-size:.75rem; margin-top:4px; opacity:0.85; }

.rec-card { background:#f8fafc; border-left:3px solid #3b82f6; padding:10px 12px; border-radius:0 8px 8px 0; margin:6px 0; font-size:.82rem; }
.rec-card.high-conf { border-left-color:#22c55e; background:#f0fdf4; }
.rec-card.med-conf { border-left-color:#eab308; background:#fefce8; }

.apply-btn { font-size:.72rem !important; padding:2px 10px !important; border-radius:6px !important; }

.insight-item { padding:6px 10px; background:#f1f5f9; border-radius:6px; margin:4px 0; font-size:.8rem; color:#334155; line-height:1.5; }

.tab-content { font-size:.82rem; }
.metric-row { margin:8px 0; }
</style>
""", unsafe_allow_html=True)


def _lazy_qa_chain():
    from qa_chain import (
        ask_question, ConversationManager,
        get_vectorstore_info, PROMPT_TEMPLATES, PROMPT_LABELS,
    )
    return ask_question, ConversationManager, get_vectorstore_info, PROMPT_TEMPLATES, PROMPT_LABELS


def _lazy_feedback():
    try:
        from feedback_engine import (
            record_rating, get_stats, get_recommendation,
            check_abuse, get_qa_pairs_count, export_qa_pairs_to_docs,
            get_negative_reasons,
        )
        return record_rating, get_stats, get_recommendation, check_abuse, get_qa_pairs_count, export_qa_pairs_to_docs, get_negative_reasons
    except ImportError:
        from feedback_engine import (
            record_rating, get_stats, get_recommendation,
            check_abuse, get_qa_pairs_count, export_qa_pairs_to_docs,
        )
        _fallback_neg_reasons = {
            "inaccurate": "回答不准确",
            "irrelevant": "来源与问题无关",
            "incomplete": "回答不完整",
            "format_bad": "格式/排版混乱",
            "other": "其他原因",
        }
        return record_rating, get_stats, get_recommendation, check_abuse, get_qa_pairs_count, export_qa_pairs_to_docs, lambda: _fallback_neg_reasons


def _invalidate_cache():
    for key in ("_cached_stats", "_cached_vs", "_cached_qa_count", "_cached_rec"):
        if key in st.session_state:
            del st.session_state[key]


@st.cache_data(ttl=60)
def _cached_vs():
    _, _, get_vi, _, _ = _lazy_qa_chain()
    return get_vi()


def _cached_stats():
    _, gs, _, _, _, _, _ = _lazy_feedback()
    return gs()

def _cached_qa_count():
    _, _, _, _, gqc, _, _ = _lazy_feedback()
    return gqc()

def _cached_rec(stats=None):
    _, _, gr, _, _, _, _ = _lazy_feedback()
    return gr(stats)


def render_sources(sources):
    if not sources:
        return
    seen, uniq = set(), []
    for s in sources:
        k = f"{s['source_file']}-p{s['page']}"
        if k not in seen:
            seen.add(k); uniq.append(s)
    with st.expander(f"📎 来源（{len(uniq)} 条）"):
        for s in uniq:
            ext = s.get("file_type", "").lower()
            bc = f"badge-{ext}" if ext in ("pdf","txt","md") else ""
            st.markdown(f'<span class="source-badge {bc}">{ext.upper()}</span> <b>{s["source_file"]}</b> — 第{s["page"]+1}页', unsafe_allow_html=True)
            st.markdown(f"> {s['content'][:280].replace(chr(10),'  \\n> ')}")
            st.divider()


def render_feedback(idx, question, answer):
    st.session_state.setdefault("feedback_given", {})
    st.session_state.setdefault("session_ratings_count", 0)

    agiven = st.session_state.feedback_given.get(idx)
    if agiven is not None and agiven != "__pending_reason__":
        color, icon, label = ("#16a34a","✅","有帮助") if agiven >= 1 else ("#dc2626","👎","无帮助")
        st.markdown(f'<span style="color:{color};font-size:.83rem;">{icon} 已标记为{label}</span>', unsafe_allow_html=True)
        return

    if agiven == "__pending_reason__":
        _, _, _, _, _, _, neg_reasons = _lazy_feedback()
        reasons = neg_reasons()
        reason_key = f"reason_{idx}"
        if reason_key not in st.session_state:
            st.session_state[reason_key] = None
        cols_r = st.columns([3, 1, 1])
        with cols_r[0]:
            reason = st.selectbox(
                "👎 请选择原因（可选）",
                options=["跳过"] + list(reasons.values()),
                format_func=lambda x: x,
                key=f"rs_{idx}",
                label_visibility="collapsed",
            )
        with cols_r[1]:
            if st.button("提交", key=f"rsb_{idx}", use_container_width=True):
                chosen = None
                if reason != "跳过":
                    for k, v in reasons.items():
                        if v == reason:
                            chosen = k; break
                rr, _, _, _, _, _, _ = _lazy_feedback()
                rr(question, answer, -1, {
                    "k": st.session_state.get("_lk",3),
                    "similarity_threshold": st.session_state.get("_lt"),
                    "search_type": st.session_state.get("_ls","similarity"),
                    "prompt_mode": st.session_state.get("_lp","anti_hallucination"),
                }, reason=chosen)
                st.session_state.feedback_given[idx] = -1
                st.session_state.session_ratings_count += 1
                _invalidate_cache(); st.rerun()
        with cols_r[2]:
            if st.button("取消", key=f"rsc_{idx}", use_container_width=True):
                st.session_state.feedback_given[idx] = None
                _invalidate_cache(); st.rerun()
        return

    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        if st.button("👍 有帮助", key=f"p_{idx}", help="标记此回答有帮助"):
            _, fb_check, _, _, _, _, _ = _lazy_feedback()
            allowed, msg = fb_check(question, st.session_state.session_ratings_count)
            if not allowed:
                st.session_state._fb_err = msg; st.rerun()
            else:
                rr, _, _, _, _, _, _ = _lazy_feedback()
                rr(question, answer, 1, {
                    "k": st.session_state.get("_lk",3),
                    "similarity_threshold": st.session_state.get("_lt"),
                    "search_type": st.session_state.get("_ls","similarity"),
                    "prompt_mode": st.session_state.get("_lp","anti_hallucination"),
                })
                st.session_state.feedback_given[idx] = 1
                st.session_state.session_ratings_count += 1
                _invalidate_cache(); st.rerun()
    with c2:
        if st.button("👎 无帮助", key=f"n_{idx}", help="标记此回答无帮助"):
            _, fb_check, _, _, _, _, _ = _lazy_feedback()
            allowed, msg = fb_check(question, st.session_state.session_ratings_count)
            if not allowed:
                st.session_state._fb_err = msg; st.rerun()
            else:
                st.session_state.feedback_given[idx] = "__pending_reason__"
                st.rerun()


def _apply_recommendation(rec):
    p = rec["param"]
    v = rec["value"]
    if p == "k":
        st.session_state._apply_k = int(v) if isinstance(v, (int, float)) else v
    elif p == "similarity_threshold":
        st.session_state._apply_t = v
    elif p == "search_type":
        st.session_state._apply_s = v
    elif p == "prompt_mode":
        st.session_state._apply_p = v


def _render_dashboard(stats, rec_data):
    tab1, tab2, tab3, tab4 = st.tabs(["📊 概览", "📈 趋势", "🔧 参数分析", "⚙️ 导出"])

    with tab1:
        hs = stats.get("health_score", 0)
        if hs >= 80:
            hcls, hlabel = "health-excellent", "优秀 🌟"
        elif hs >= 60:
            hcls, hlabel = "health-good", "良好 ✅"
        elif hs >= 40:
            hcls, hlabel = "health-fair", "一般 ⚠️"
        else:
            hcls, hlabel = "health-poor", "需改进 ❌"
        st.markdown(f'<div class="health-gauge {hcls}"><div class="health-score">{hs}</div><div class="health-label">知识库健康度 · {hlabel}</div></div>', unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("总反馈数", stats["total"], delta=None)
        with mc2:
            delta_color = "normal" if stats.get("positive_rate", 0) >= 50 else "inverse"
            st.metric("好评率", f"{stats.get('positive_rate', 0):.1f}%", f"{stats.get('positive', 0)}👍 / {stats.get('negative', 0)}👎", delta_color=delta_color)
        with mc3:
            qac = _cached_qa_count()
            st.metric("问答对", f"{qac} 条", "+可导出补充知识库")

        if rec_data.get("recommendations"):
            st.markdown("#### 🔝 推荐参数（一键应用）")
            for rec in rec_data["recommendations"]:
                conf_cls = "high-conf" if rec.get("confidence") == "high" else "med-conf"
                conf_tag = "高置信" if rec.get("confidence") == "high" else "收集中"
                param_labels = {
                    "k": f'k={rec["value"]}',
                    "similarity_threshold": f'阈值={"关闭" if rec["value"] is None else rec["value"]}',
                    "search_type": {"similarity":"相似度检索","mmr":"MMR多样性"}.get(rec["value"], rec["value"]),
                    "prompt_mode": f'模式={rec["value"][:8]}',
                }
                plabel = param_labels.get(rec["param"], rec["param"])
                rc1, rc2 = st.columns([5, 1])
                with rc1:
                    st.markdown(
                        f'<div class="rec-card {conf_cls}">'
                        f'<b>{plabel}</b> · Wilson={rec["wilson_score"]:.2f} '
                        f'· 好评率{rec["raw_positive_rate"]:.0%}({rec["sample_size"]}条) '
                        f'<span style="font-size:.68rem;opacity:0.6;">[{conf_tag}]</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with rc2:
                    if st.button("✨ 应用", key=f"app_{rec['param']}", use_container_width=True, **{"css_classes": ["apply-btn"]} if False else {}):
                        _apply_recommendation(rec)
                        st.rerun()
        else:
            if rec_data.get("has_data"):
                st.info("正在分析反馈数据，继续收集后将显示推荐...")
            else:
                st.caption("💡 回答问题并点击 👍/👎 后，此处将显示智能推荐")

        if rec_data.get("insights"):
            st.markdown("#### 💡 智能洞察")
            for ins in rec_data["insights"]:
                st.markdown(f'<div class="insight-item">• {ins}</div>', unsafe_allow_html=True)

    with tab2:
        timeline = stats.get("timeline", [])
        if timeline:
            df_data = {"时间段": [], "反馈数": [], "好评率(%)": []}
            active_buckets = [t for t in timeline if t["total"] > 0]
            if len(active_buckets) >= 2:
                for t in active_buckets:
                    df_data["时间段"].append(t["bucket"])
                    df_data["反馈数"].append(t["total"])
                    df_data["好评率(%)"].append(t["rate"] if t["rate"] is not None else 0)
                st.dataframe(df_data, use_container_width=True, hide_index=True)
                rates_only = [t["rate"] for t in active_buckets if t["rate"] is not None]
                if rates_only:
                    chart_data = {"好评率": rates_only}
                    row_idx = list(range(len(rates_only)))
                    st.line_chart(chart_data, height=200, use_container_width=True)
            else:
                st.caption("需要更多时间跨度的数据来绘制趋势图（至少2个时间点有数据）")
        else:
            st.caption("暂无趋势数据")

    with tab3:
        by_k = stats.get("by_k", {})
        by_thresh = stats.get("by_threshold", {})
        by_search = stats.get("by_search_type", {})
        by_prompt = stats.get("by_prompt_mode", {})
        by_reason = stats.get("by_reason", {})

        has_param_data = any([by_k, by_thresh, by_search, by_prompt])
        if has_param_data:
            sc_a, sc_b = st.columns(2)

            with sc_a:
                if by_k:
                    st.markdown("**检索数量 k**")
                    k_df = {"k值": list(by_k.keys()), "总数": [by_k[k]["total"] for k in by_k], "好评数": [by_k[k]["positive"] for k in by_k]}
                    st.dataframe(k_df, use_container_width=True, hide_index=True)
                    k_rates = {k: round(by_k[k]["positive"]/by_k[k]["total"]*100, 1) if by_k[k]["total"] > 0 else 0 for k in by_k}
                    st.bar_chart(k_rates, height=150, horizontal=False)

                if by_thresh:
                    st.markdown("**拒答阈值**")
                    t_labels = {"off": "关闭"}
                    t_df = {"阈值": [t_labels.get(t, t) for t in by_thresh.keys()], "总数": [by_thresh[t]["total"] for t in by_thresh], "好评数": [by_thresh[t]["positive"] for t in by_thresh]}
                    st.dataframe(t_df, use_container_width=True, hide_index=True)
                    t_rates = {t_labels.get(t, t): round(by_thresh[t]["positive"]/by_thresh[t]["total"]*100, 1) if by_thresh[t]["total"] > 0 else 0 for t in by_thresh}
                    st.bar_chart(t_rates, height=150, horizontal=False)

            with sc_b:
                if by_search:
                    st.markdown("**检索策略**")
                    s_labels = {"similarity": "相似度", "mmr": "MMR多样性"}
                    s_df = {"策略": [s_labels.get(s, s) for s in by_search.keys()], "总数": [by_search[s]["total"] for s in by_search], "好评数": [by_search[s]["positive"] for s in by_search]}
                    st.dataframe(s_df, use_container_width=True, hide_index=True)
                    s_rates = {s_labels.get(s, s): round(by_search[s]["positive"]/by_search[s]["total"]*100, 1) if by_search[s]["total"] > 0 else 0 for s in by_search}
                    st.bar_chart(s_rates, height=150, horizontal=False)

                if by_prompt:
                    st.markdown("**提示词模式**")
                    p_df = {"模式": list(by_prompt.keys()), "总数": [by_prompt[p]["total"] for p in by_prompt], "好评数": [by_prompt[p]["positive"] for p in by_prompt]}
                    st.dataframe(p_df, use_container_width=True, hide_index=True)
                    p_rates = {p[:10]: round(by_prompt[p]["positive"]/by_prompt[p]["total"]*100, 1) if by_prompt[p]["total"] > 0 else 0 for p in by_prompt}
                    st.bar_chart(p_rates, height=150, horizontal=False)

            if by_reason:
                st.markdown("---")
                st.markdown("**👎 差评原因分布**")
                _, _, _, _, neg_reasons_fn = _lazy_feedback()
                rmap = neg_reasons_fn()
                r_labels = [rmap.get(r, r) for r in by_reason.keys()]
                r_counts = list(by_reason.values())
                r_total = sum(r_counts)
                r_pcts = [round(c/r_total*100, 1) if r_total > 0 else 0 for c in r_counts]
                reason_df = {"原因": r_labels, "数量": r_counts, "占比(%)": r_pcts}
                st.dataframe(reason_df, use_container_width=True, hide_index=True)
                st.bar_chart(dict(zip(r_labels, r_counts)), height=120, horizontal=False)
        else:
            st.caption("切换不同参数组合使用系统后，此处将展示对比分析")

    with tab4:
        qac = _cached_qa_count()
        st.markdown(f"#### 📦 高分问答对：**{qac}** 条")
        if qac > 0:
            st.info("这些问答对来自用户好评，可导出后通过 `python ingest.py --incremental` 补充到知识库，实现自增长")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                if st.button("📤 导出为 Markdown", use_container_width=True, type="primary"):
                    _, _, _, _, _, ep_fn, _ = _lazy_feedback()
                    count = ep_fn()
                    if count > 0:
                        st.success(f"已导出 {count} 条 → docs/用户反馈问答对.md")
                    else:
                        st.warning("导出失败")
            with col_e2:
                st.code("python ingest.py --incremental", language="bash")
            st.caption("💡 导出的 Markdown 文件支持自动去重，同一问题的多次好评只保留最高分版本")
        else:
            st.caption("暂无高分问答对，对回答点击 👍 即可收集")


with st.sidebar:
    st.markdown("### ⚙️ 参数设置")

    apply_k = st.session_state.pop("_apply_k", None)
    apply_t = st.session_state.pop("_apply_t", None)
    apply_s = st.session_state.pop("_apply_s", None)
    apply_p = st.session_state.pop("_apply_p", None)

    default_k = apply_k if apply_k is not None else 3
    k = st.slider("**检索数量 k**", 1, 10, int(default_k), help="越大回答越详细")

    default_s = apply_s if apply_s else "similarity"
    search_type = st.selectbox("**检索策略**", ["similarity","mmr"],
                                 index=0 if default_s == "similarity" else 1,
                                 format_func=lambda x: {"similarity":"相似度检索","mmr":"MMR多样性"}[x])

    _, _, _, _, PL = _lazy_qa_chain()
    prompt_keys = list(PL.keys())
    default_p = apply_p if apply_p else prompt_keys[0] if prompt_keys else "anti_hallucination"
    p_idx = prompt_keys.index(default_p) if default_p in prompt_keys else 0
    prompt_mode = st.selectbox(
        "**提示词模式**", prompt_keys,
        index=p_idx,
        format_func=lambda x: PL[x],
    )

    enable_memory = st.checkbox("**多轮对话记忆**", True)

    with st.expander("🔬 进阶参数", expanded=False):
        default_t_val = float(apply_t) if apply_t is not None else 0.0
        similarity_threshold = st.slider("A3·拒答阈值", 0.0, 1.0, default_t_val, 0.05, help="低于此值返回无法回答(0关闭)")
        threshold_val = similarity_threshold if similarity_threshold > 0 else None
        if search_type == "mmr":
            mmr_fetch_k = st.slider("B1·fetch_k", 5, 50, 15)
            mmr_lambda = st.slider("B1·lambda_mult", 0.0, 1.0, 0.5, 0.05)
        else:
            mmr_fetch_k, mmr_lambda = None, 0.5
        memory_window_val = st.slider("B3·记忆窗口(轮)", 0, 20, 0) if enable_memory else 0
        win_val = memory_window_val if memory_window_val > 0 else None

    st.divider()

    vs = _cached_vs()
    stats = _cached_stats()

    st.markdown("#### 📊 数据状态")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.metric("向量片段", vs["count"], delta=None, delta_color="off")
    with sc2:
        if stats["total"] > 0:
            st.metric("反馈数", stats["total"], f"{stats['positive_rate']}% 👍", delta_color="normal" if stats['positive_rate']>=50 else "inverse")

    if vs["files"]:
        with st.expander(f"📄 已加载文档（{len(vs['files'])} 个）"):
            for f in vs["files"]:
                ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
                bc = f"badge-{ext}" if ext in ("pdf","txt","md") else ""
                st.markdown(f'<span class="source-badge {bc}">{ext.upper()}</span> {f}', unsafe_allow_html=True)

    st.divider()

    if stats["total"] > 0:
        rec_data = _cached_rec(stats)
        with st.expander("🧬 自进化仪表盘", expanded=(stats["total"] > 0)):
            _render_dashboard(stats, rec_data)
    else:
        st.caption("💡 回答问题后点击 👍/👎 开始收集反馈")

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        if "conv_mgr" in st.session_state:
            st.session_state.conv_mgr.reset()
        st.rerun()


st.title("📚 MyLibrary RAG 智能文档问答")
st.caption("基于 RAG 技术的本地知识库问答系统 · 支持 Python / LangChain 文档智能检索 · 自进化反馈引擎 v2.0")

if vs["count"] == 0:
    st.warning("⚠️ 向量数据库为空，请先运行 `python ingest.py` 导入文档")
    st.code("python ingest.py", language="bash")
    st.stop()

if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ 未检测到 API Key，请在 `.env` 文件中配置 `OPENAI_API_KEY`")
    st.code("OPENAI_API_KEY=your_api_key_here\nOPENAI_BASE_URL=https://api.deepseek.com/v1\nMODEL_NAME=deepseek-chat", language="bash")
    st.stop()

if st.session_state.get("_fb_err"):
    st.warning(st.session_state._fb_err)
    st.session_state._fb_err = None

ask_qa, ConvMgr, _, PT, PL = _lazy_qa_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conv_mgr" not in st.session_state:
    st.session_state.conv_mgr = ConvMgr(
        similarity_threshold=threshold_val, memory_window=win_val,
        fetch_k=mmr_fetch_k, lambda_mult=mmr_lambda,
    )

st.session_state._lk = k
st.session_state._lt = threshold_val
st.session_state._ls = search_type
st.session_state._lp = prompt_mode

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            mode = msg.get("mode", "rag")
            mode_labels = {"rag": "📚 RAG检索", "chat": "💬 自由对话", "reject": "⛔ 已拒答"}
            mode_cls = f"mode-{mode}"
            st.markdown(f'<span class="mode-badge {mode_cls}">{mode_labels.get(mode, mode)}</span>', unsafe_allow_html=True)
            if msg.get("sources"): render_sources(msg["sources"])
            render_feedback(idx, msg.get("_q",""), msg["content"])

if not st.session_state.messages:
    suggestions = ["RAG技术的核心原理是什么？","LangChain框架有哪些主要组件？","Python在AI开发中有哪些应用？"]
    st.markdown(f"""
    <div class="welcome-box">
        <h2>👋 欢迎使用 MyLibrary RAG</h2>
        <p>基于 <strong>RAG（检索增强生成）</strong> + <strong>DeepSeek API</strong> 构建，已加载 Python / LangChain / Streamlit 技术文档。<br>
        📚 文档相关问题自动检索知识库回答 · 💬 日常对话直接与 AI 聊天 · 支持多轮对话与来源追溯<br>
        🧬 自进化引擎：Wilson智能推荐 + 参数一键应用 + 差评分析 + 知识库自增长</p>
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        if cols[i].button(s, key=f"s_{i}", use_container_width=True):
            st.session_state._pq = s; st.rerun()

if "_pq" in st.session_state and st.session_state._pq:
    prompt = st.session_state._pq
    st.session_state._pq = None
elif prompt := st.chat_input("输入您的问题..."):
    pass
else:
    prompt = None

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        with st.spinner("🔍 思考中..."):
            try:
                cm = st.session_state.conv_mgr
                cm.k = k; cm.prompt_mode = prompt_mode; cm.search_type = search_type
                cm.similarity_threshold = threshold_val; cm.memory_window = win_val
                cm.fetch_k = mmr_fetch_k; cm.lambda_mult = mmr_lambda
                cm._chain = None

                if enable_memory:
                    ans, src, mode = cm.ask(prompt)
                else:
                    ans, src, mode = ask_qa(question=prompt, k=k, prompt_mode=prompt_mode,
                                      search_type=search_type, similarity_threshold=threshold_val)

                st.markdown(ans)
                if mode == "rag":
                    render_sources(src)
                render_feedback(len(st.session_state.messages), prompt, ans)

                st.session_state.messages.append({"role":"assistant","content":ans,"sources":src,"_q":prompt,"mode":mode})
            except Exception as e:
                err = f"⚠️ 出错: {e}"
                st.error(err)
                st.session_state.messages.append({"role":"assistant","content":err})
