import os
import sys
import re
import logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylibs"))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from config import (
    CHAT_FALLBACK_THRESHOLD, CHAT_DOMAIN_THRESHOLD, DEFAULT_K,
    DEFAULT_PROMPT_MODE, DEFAULT_SEARCH_TYPE,
)
from vectorstore import get_retriever, search_with_scores
from llm import get_llm, chat_directly, condense_question

logger = logging.getLogger(__name__)

ANTI_HALLUCINATION_TEMPLATE = """你是一个专业的图书管理员，请用严谨的学术风格回答问题。请根据以下上下文回答问题。如果上下文中没有包含回答该问题所需的信息，请直接回答"根据提供的文档内容，无法回答该问题"，不要编造或推测任何内容。

上下文：
{context}

问题：{question}

回答："""

SIMPLE_TEMPLATE = """请根据以下上下文回答问题：
{context}

问题：{question}

回答："""

ACADEMIC_TEMPLATE = """你是一位严谨的学术图书管理员，擅长从文献中提取关键信息并进行综合分析。请基于以下参考资料回答问题，要求：
1. 回答需有据可依，引用相关文档内容
2. 如果信息不足，明确指出并说明缺少什么
3. 使用规范的学术语言，逻辑清晰

参考资料：
{context}

问题：{question}

回答："""

CROSS_DOC_TEMPLATE = """请基于以下来自多篇文档的参考资料，进行跨文档对比分析。要求：
1. 指出不同文档中的相同观点和不同观点
2. 综合多篇文档的信息给出完整回答
3. 标注信息来源

参考资料：
{context}

问题：{question}

跨文档分析："""

PROMPT_TEMPLATES = {
    "anti_hallucination": ANTI_HALLUCINATION_TEMPLATE,
    "simple": SIMPLE_TEMPLATE,
    "academic": ACADEMIC_TEMPLATE,
    "cross_doc": CROSS_DOC_TEMPLATE,
}

PROMPT_LABELS = {
    "anti_hallucination": "🛡️ 抑制幻觉",
    "simple": "📝 简单模式",
    "academic": "🎓 学术风格",
    "cross_doc": "🔍 跨文档对比",
}

CHAT_PATTERNS = [
    r"^(你好|您好|嗨|hi|hello|hey|早|早上好|下午好|晚上好)[\s!！?？,，。.]*$",
    r"^(你是谁|你是什么|你叫什么|介绍一下[你自]你|你能做什|你能帮|你是干)",
    r"^(谢谢|感谢|再见|拜拜|goodbye|bye)[\s!！?？]*$",
    r"^(天气|讲个笑话|无聊|你好啊|在吗|有人吗)",
    r"^[\s!！?？。，,]+$",
]

DOMAIN_KEYWORDS = {
    "rag", "检索", "向量", "嵌入", "embedding", "langchain", "python",
    "streamlit", "chromadb", "大模型", "llm", "提示词", "prompt",
    "文档", "知识库", "分块", "chunk", "记忆", "memory", "chain",
    "agent", "retriever", "transformer", "torch", "模型", "框架",
    "组件", "api", "函数", "编程", "开发", "数据", "机器学习",
    "深度学习", "nlp", "自然语言", "对话", "生成", "微调",
}

RAG_EMPTY_PATTERNS = [
    "无法回答", "无法提供", "无法找到", "没有包含",
    "文档内容中未", "上下文中没有", "提供的文档",
    "知识库中没有", "检索到的文档", "根据提供的文档内容，无法",
]


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _extract_sources(source_docs):
    sources = []
    for doc in source_docs:
        sources.append({
            "content": doc.page_content[:300],
            "source_file": doc.metadata.get("source_file", "未知"),
            "page": doc.metadata.get("page", 0),
            "file_type": doc.metadata.get("file_type", "未知"),
        })
    return sources


def _looks_like_chat_question(question):
    q = question.strip()
    if len(q) <= 1:
        return True
    for pat in CHAT_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return True
    if len(q) <= 6 and not any(kw in q.lower() for kw in DOMAIN_KEYWORDS):
        return True
    return False


def _should_fallback_to_chat(max_score, question):
    if max_score < CHAT_FALLBACK_THRESHOLD:
        if not any(kw in question.lower() for kw in DOMAIN_KEYWORDS):
            return True
    if max_score < CHAT_DOMAIN_THRESHOLD:
        if not any(kw in question.lower() for kw in DOMAIN_KEYWORDS):
            return True
    return False


def _is_rag_empty_answer(answer):
    return any(p in answer for p in RAG_EMPTY_PATTERNS)


def _rag_query(question, k=3, prompt_mode="anti_hallucination",
               search_type="similarity", fetch_k=None, lambda_mult=0.5):
    retriever = get_retriever(k=k, search_type=search_type, fetch_k=fetch_k, lambda_mult=lambda_mult)
    llm = get_llm()
    template = PROMPT_TEMPLATES.get(prompt_mode, ANTI_HALLUCINATION_TEMPLATE)
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    source_docs = retriever.invoke(question)
    context = _format_docs(source_docs)
    answer = chain.invoke({"context": context, "question": question})
    sources = _extract_sources(source_docs)
    return answer, sources


def ask_question(question, k=3, prompt_mode="anti_hallucination", search_type="similarity",
                 similarity_threshold=None, fetch_k=None, lambda_mult=0.5):
    if _looks_like_chat_question(question):
        answer = chat_directly(question)
        return answer, [], "chat"

    docs, scores = search_with_scores(
        question, k=k, search_type=search_type,
        fetch_k=fetch_k, lambda_mult=lambda_mult,
    )

    if not docs or not scores:
        answer = chat_directly(question)
        return answer, [], "chat"

    max_score = max(scores)

    if similarity_threshold and similarity_threshold > 0:
        if max_score < similarity_threshold:
            reject_msg = (
                f"⚠️ 根据提供的文档内容，无法回答该问题。\n\n"
                f"（检索到的文档片段与问题的最高相似度为 "
                f"{max_score:.2f}，"
                f"低于设定阈值 {similarity_threshold}）\n\n"
                f"建议您尝试换个方式提问，或确认文档中是否包含相关信息。"
            )
            return reject_msg, [], "reject"

    if _should_fallback_to_chat(max_score, question):
        answer = chat_directly(question)
        return answer, [], "chat"

    rag_answer, sources = _rag_query(
        question, k=k, prompt_mode=prompt_mode,
        search_type=search_type, fetch_k=fetch_k, lambda_mult=lambda_mult,
    )

    if _is_rag_empty_answer(rag_answer) and max_score < 0.55:
        answer = chat_directly(question)
        return answer, [], "chat"

    return rag_answer, sources, "rag"


class ConversationManager:
    def __init__(self, k=3, prompt_mode="anti_hallucination", search_type="similarity",
                 similarity_threshold=None, memory_window=None,
                 fetch_k=None, lambda_mult=0.5):
        self.k = k
        self.prompt_mode = prompt_mode
        self.search_type = search_type
        self.similarity_threshold = similarity_threshold
        self.memory_window = memory_window
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.chat_history = []

    def _trim_history(self):
        if self.memory_window and self.memory_window > 0:
            max_msgs = self.memory_window * 2
            if len(self.chat_history) > max_msgs:
                self.chat_history = self.chat_history[-max_msgs:]

    def _add_to_history(self, question, answer):
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        self._trim_history()

    def _handle_chat(self, question):
        answer = chat_directly(question, self.chat_history)
        self._add_to_history(question, answer)
        return answer, [], "chat"

    def ask(self, question):
        if _looks_like_chat_question(question):
            return self._handle_chat(question)

        docs, scores = search_with_scores(
            question, k=self.k, search_type=self.search_type,
            fetch_k=self.fetch_k, lambda_mult=self.lambda_mult,
        )

        if not docs or not scores:
            return self._handle_chat(question)

        max_score = max(scores)

        if self.similarity_threshold is not None and self.similarity_threshold > 0:
            if max_score < self.similarity_threshold:
                reject_msg = (
                    f"⚠️ 根据提供的文档内容，无法回答该问题。\n\n"
                    f"（检索到的文档片段与问题的最高相似度为 "
                    f"{max_score:.2f}，"
                    f"低于设定阈值 {self.similarity_threshold}）\n\n"
                    f"建议您尝试换个方式提问，或确认文档中是否包含相关信息。"
                )
                self._add_to_history(question, reject_msg)
                return reject_msg, [], "reject"

        if _should_fallback_to_chat(max_score, question):
            return self._handle_chat(question)

        llm = get_llm()
        condensed = condense_question(llm, question, self.chat_history)

        rag_answer, sources = _rag_query(
            condensed, k=self.k, prompt_mode=self.prompt_mode,
            search_type=self.search_type, fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )

        if _is_rag_empty_answer(rag_answer):
            return self._handle_chat(question)

        self._add_to_history(question, rag_answer)
        return rag_answer, sources, "rag"

    def reset(self):
        self.chat_history = []


def cli_qa():
    print("=" * 60)
    print("MyLibrary RAG - 命令行问答模式")
    print("输入 'quit' 或 'exit' 退出")
    print("命令: /k <数字>     设置检索数量")
    print("      /mode <模式>   设置提示词模式")
    print("      /search <类型> 设置检索策略(similarity/mmr)")
    print("      /threshold <值> 设置拒答阈值(0=关闭)")
    print("      /window <数字> 设置记忆窗口(0=不限制)")
    print("=" * 60)

    k = DEFAULT_K
    prompt_mode = DEFAULT_PROMPT_MODE
    search_type = DEFAULT_SEARCH_TYPE
    similarity_threshold = None
    memory_window = None
    conv_mgr = ConversationManager(
        k=k, prompt_mode=prompt_mode, search_type=search_type,
        similarity_threshold=similarity_threshold, memory_window=memory_window,
    )

    thresh_str = f"{similarity_threshold}" if similarity_threshold is not None else "关闭"
    win_str = f"{memory_window}" if memory_window is not None else "不限制"
    print(f"当前设置: k={k}, 模式={prompt_mode}, 检索={search_type}, 阈值={thresh_str}, 窗口={win_str}")
    print()

    while True:
        try:
            question = input("请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not question:
            continue

        if question.startswith("/k "):
            try:
                k = int(question.split()[1])
                conv_mgr.k = k
                print(f"检索数量已设置为 k={k}")
            except (IndexError, ValueError):
                print("用法: /k <数字>")
            continue

        if question.startswith("/mode "):
            mode = question.split(maxsplit=1)
            if len(mode) > 1 and mode[1] in PROMPT_TEMPLATES:
                prompt_mode = mode[1]
                conv_mgr.prompt_mode = prompt_mode
                print(f"提示词模式已设置为: {prompt_mode}")
            else:
                print(f"可用模式: {', '.join(PROMPT_TEMPLATES.keys())}")
            continue

        if question.startswith("/search "):
            st_val = question.split(maxsplit=1)
            if len(st_val) > 1 and st_val[1] in ("similarity", "mmr"):
                search_type = st_val[1]
                conv_mgr.search_type = search_type
                print(f"检索策略已设置为: {search_type}")
            else:
                print("可用策略: similarity, mmr")
            continue

        if question.startswith("/threshold "):
            try:
                val = float(question.split()[1])
                if val <= 0:
                    similarity_threshold = None
                    print("拒答阈值已关闭")
                else:
                    similarity_threshold = val
                    print(f"拒答阈值已设置为: {val}")
                conv_mgr.similarity_threshold = similarity_threshold
            except (IndexError, ValueError):
                print("用法: /threshold <数值> (0=关闭, 推荐0.3/0.5/0.7)")
            continue

        if question.startswith("/window "):
            try:
                val = int(question.split()[1])
                if val <= 0:
                    memory_window = None
                    print("记忆窗口已取消限制")
                else:
                    memory_window = val
                    print(f"记忆窗口已设置为: {val} 轮")
                conv_mgr.memory_window = memory_window
            except (IndexError, ValueError):
                print("用法: /window <轮数> (0=不限制)")
            continue

        try:
            answer, sources, mode = conv_mgr.ask(question)
            print(f"\n回答: {answer}")
            if sources:
                print(f"\n来源文档:")
                seen = set()
                for s in sources:
                    key = f"{s['source_file']}-p{s['page']}"
                    if key not in seen:
                        seen.add(key)
                        print(f"  - [{s['file_type'].upper()}] {s['source_file']} (第{s['page']+1}页)")
            print()
        except Exception as e:
            print(f"错误: {e}\n")
