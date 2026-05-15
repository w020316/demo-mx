import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from config import CHROMA_DIR, PROJECT_ROOT
from embeddings import ChromaDefaultEmbeddings

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DEFAULT_PROMPT_TEMPLATE = """你是一个专业的图书管理员，请用严谨的学术风格回答问题。请根据以下上下文回答问题。如果上下文中没有包含回答该问题所需的信息，请直接回答"根据提供的文档内容，无法回答该问题"，不要编造或推测任何内容。

上下文：
{context}

问题：{question}

回答："""

SIMPLE_PROMPT_TEMPLATE = """请根据以下上下文回答问题：
{context}

问题：{question}

回答："""

ACADEMIC_PROMPT_TEMPLATE = """你是一位严谨的学术图书管理员，擅长从文献中提取关键信息并进行综合分析。请基于以下参考资料回答问题，要求：
1. 回答需有据可依，引用相关文档内容
2. 如果信息不足，明确指出并说明缺少什么
3. 使用规范的学术语言，逻辑清晰

参考资料：
{context}

问题：{question}

回答："""

CROSS_DOC_PROMPT_TEMPLATE = """请基于以下来自多篇文档的参考资料，进行跨文档对比分析。要求：
1. 指出不同文档中的相同观点和不同观点
2. 综合多篇文档的信息给出完整回答
3. 标注信息来源

参考资料：
{context}

问题：{question}

跨文档分析："""

CONDENSE_TEMPLATE = """根据以下对话历史和最新问题，将问题重写为一个独立的、完整的问题。

对话历史：
{chat_history}

最新问题：{question}

独立问题："""

PROMPT_TEMPLATES = {
    "anti_hallucination": DEFAULT_PROMPT_TEMPLATE,
    "simple": SIMPLE_PROMPT_TEMPLATE,
    "academic": ACADEMIC_PROMPT_TEMPLATE,
    "cross_doc": CROSS_DOC_PROMPT_TEMPLATE,
}

PROMPT_LABELS = {
    "anti_hallucination": "🛡️ 抑制幻觉",
    "simple": "📝 简单模式",
    "academic": "🎓 学术风格",
    "cross_doc": "🔍 跨文档对比",
}

_embeddings_instance = None


def _get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = ChromaDefaultEmbeddings()
    return _embeddings_instance


def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_get_embeddings(),
    )


def get_vectorstore_info():
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        metadatas = vs._collection.get(include=["metadatas"])["metadatas"]
        files = set()
        for m in metadatas:
            if m and "source_file" in m:
                files.add(m["source_file"])
        return {"count": count, "files": sorted(files)}
    except Exception:
        return {"count": 0, "files": []}


def get_retriever(k=3, search_type="similarity", fetch_k=None, lambda_mult=0.5):
    vectorstore = get_vectorstore()
    if search_type == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k or k * 5,
                "lambda_mult": lambda_mult,
            },
        )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def search_with_scores(question, k=3, search_type="similarity", fetch_k=None, lambda_mult=0.5):
    vectorstore = get_vectorstore()
    try:
        if search_type == "mmr":
            raw_results = vectorstore.max_marginal_relevance_search_with_score(
                question, k=k, fetch_k=fetch_k or k * 5, lambda_mult=lambda_mult,
            )
        else:
            raw_results = vectorstore.similarity_search_with_score(question, k=k)
        docs = []
        scores = []
        for item in raw_results:
            if search_type == "mmr" and len(item) == 2:
                doc, distance = item
            else:
                doc, distance = item
            docs.append(doc)
            similarity = 1.0 / (1.0 + distance)
            scores.append(similarity)
        return docs, scores
    except Exception:
        return [], []


def get_llm(temperature=0):
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
    )


def _build_prompt(prompt_mode):
    template = PROMPT_TEMPLATES.get(prompt_mode, DEFAULT_PROMPT_TEMPLATE)
    return PromptTemplate(template=template, input_variables=["context", "question"])


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


def _condense_question(llm, question, chat_history_messages):
    if not chat_history_messages:
        return question
    history_text = ""
    for msg in chat_history_messages[-10:]:
        content = str(msg.content).replace("{", "{{").replace("}", "}}")
        if isinstance(msg, HumanMessage):
            history_text += f"用户: {content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"助手: {content}\n"
    if not history_text.strip():
        return question
    safe_q = question.replace("{", "{{").replace("}", "}}")
    prompt = CONDENSE_TEMPLATE.format(chat_history=history_text, question=safe_q)
    try:
        response = llm.invoke(prompt)
        condensed = response.content.strip()
        return condensed if condensed else question
    except Exception:
        return question


def get_qa_chain(k=3, prompt_mode="anti_hallucination", search_type="similarity",
                 fetch_k=None, lambda_mult=0.5):
    retriever = get_retriever(k=k, search_type=search_type, fetch_k=fetch_k, lambda_mult=lambda_mult)
    llm = get_llm()
    prompt = _build_prompt(prompt_mode)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


import re

_CHAT_PATTERNS = [
    r"^(你好|您好|嗨|hi|hello|hey|早|早上好|下午好|晚上好)[\s!！?？,，。.]*$",
    r"^(你是谁|你是什么|你叫什么|介绍一下[你自]你|你能做什|你能帮|你是干)",
    r"^(谢谢|感谢|再见|拜拜|goodbye|bye)[\s!！?？]*$",
    r"^(天气|讲个笑话|无聊|你好啊|在吗|有人吗)",
    r"^[\s!！?？。，,]+$",
]

_DOMAIN_KEYWORDS = {
    "rag", "检索", "向量", "嵌入", "embedding", "langchain", "python",
    "streamlit", "chromadb", "大模型", "llm", "提示词", "prompt",
    "文档", "知识库", "分块", "chunk", "记忆", "memory", "chain",
    "agent", "retriever", "transformer", "torch", "模型", "框架",
    "组件", "api", "函数", "编程", "开发", "数据", "机器学习",
    "深度学习", "nlp", "自然语言", "对话", "生成", "微调",
}

CHAT_FALLBACK_THRESHOLD = 0.42
CHAT_DOMAIN_THRESHOLD = 0.55

CHAT_SYSTEM_PROMPT = """你是 MyLibrary RAG 智能文档问答系统的助手。你的职责：
1. 当用户问关于知识库文档中的内容时，基于检索到的文档回答（由系统自动处理）
2. 当用户进行日常问候、闲聊或问非文档问题时，友好地直接回答
3. 自我介绍时说明：你是一个基于 RAG（检索增强生成）技术构建的本地知识库问答系统，使用 DeepSeek 大模型驱动，支持 Python、LangChain、Streamlit 等技术文档的智能检索与问答。
4. 回答风格：简洁、专业、有帮助。中文回答。"""


def _looks_like_chat_question(question):
    q = question.strip()
    if len(q) <= 1:
        return True
    for pat in _CHAT_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return True
    if len(q) <= 6 and not any(kw in q.lower() for kw in _DOMAIN_KEYWORDS):
        return True
    return False


def _chat_directly(question, chat_history_messages=None):
    llm = get_llm(temperature=0.7)
    messages = [HumanMessage(content=CHAT_SYSTEM_PROMPT)]
    if chat_history_messages:
        messages.extend(chat_history_messages[-10:])
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content


def ask_question(question, k=3, prompt_mode="anti_hallucination", search_type="similarity",
                 similarity_threshold=None, fetch_k=None, lambda_mult=0.5):
    if _looks_like_chat_question(question):
        answer = _chat_directly(question)
        return answer, [], "chat"

    docs, scores = search_with_scores(question, k=k, search_type=search_type,
                                      fetch_k=fetch_k, lambda_mult=lambda_mult)

    if not docs or scores is None or not scores:
        answer = _chat_directly(question)
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

    if max_score < CHAT_FALLBACK_THRESHOLD:
        q_lower = question.lower()
        if not any(kw in q_lower for kw in _DOMAIN_KEYWORDS):
            answer = _chat_directly(question)
            return answer, [], "chat"

    if max_score < CHAT_DOMAIN_THRESHOLD:
        q_lower = question.lower()
        if not any(kw in q_lower for kw in _DOMAIN_KEYWORDS):
            answer = _chat_directly(question)
            return answer, [], "chat"

    qa_chain = get_qa_chain(k=k, prompt_mode=prompt_mode, search_type=search_type,
                            fetch_k=fetch_k, lambda_mult=lambda_mult)
    result = qa_chain.invoke({"query": question})
    return result["result"], _extract_sources(result.get("source_documents", [])), "rag"


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
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        self._chain = None
        self._last_params = None

    def _get_chain(self):
        param_key = (self.k, self.prompt_mode, self.search_type, self.fetch_k, self.lambda_mult)
        if self._chain is not None and self._last_params == param_key:
            return self._chain

        retriever = get_retriever(
            k=self.k, search_type=self.search_type,
            fetch_k=self.fetch_k, lambda_mult=self.lambda_mult,
        )
        llm = get_llm()
        qa_prompt = _build_prompt(self.prompt_mode)

        self._chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt},
        )
        self._last_params = param_key
        return self._chain

    def ask(self, question):
        if _looks_like_chat_question(question):
            llm = get_llm(temperature=0.7)
            chat_history = self.memory.chat_memory.messages
            answer = _chat_directly(question, chat_history)
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            if self.memory_window:
                history = self.memory.chat_memory.messages
                if len(history) > self.memory_window * 2:
                    trimmed = history[-(self.memory_window * 2):]
                    self.memory.chat_memory.messages = trimmed
            return answer, [], "chat"

        docs, scores = search_with_scores(
            question, k=self.k, search_type=self.search_type,
            fetch_k=self.fetch_k, lambda_mult=self.lambda_mult,
        )

        if not docs or scores is None or not scores:
            llm = get_llm(temperature=0.7)
            chat_history = self.memory.chat_memory.messages
            answer = _chat_directly(question, chat_history)
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            if self.memory_window:
                history = self.memory.chat_memory.messages
                if len(history) > self.memory_window * 2:
                    trimmed = history[-(self.memory_window * 2):]
                    self.memory.chat_memory.messages = trimmed
            return answer, [], "chat"

        max_score = max(scores)

        if self.similarity_threshold is not None and self.similarity_threshold > 0:
            if max_score < self.similarity_threshold:
                max_s = max(scores) if scores else 0
                reject_msg = (
                    f"⚠️ 根据提供的文档内容，无法回答该问题。\n\n"
                    f"（检索到的文档片段与问题的最高相似度为 "
                    f"{max_s:.2f}，"
                    f"低于设定阈值 {self.similarity_threshold}）\n\n"
                    f"建议您尝试换个方式提问，或确认文档中是否包含相关信息。"
                )
                self.memory.chat_memory.add_user_message(question)
                self.memory.chat_memory.add_ai_message(reject_msg)
                return reject_msg, [], "reject"

        if max_score < CHAT_FALLBACK_THRESHOLD:
            q_lower = question.lower()
            if not any(kw in q_lower for kw in _DOMAIN_KEYWORDS):
                llm = get_llm(temperature=0.7)
                chat_history = self.memory.chat_memory.messages
                answer = _chat_directly(question, chat_history)
                self.memory.chat_memory.add_user_message(question)
                self.memory.chat_memory.add_ai_message(answer)
                if self.memory_window:
                    history = self.memory.chat_memory.messages
                    if len(history) > self.memory_window * 2:
                        trimmed = history[-(self.memory_window * 2):]
                        self.memory.chat_memory.messages = trimmed
                return answer, [], "chat"

        if max_score < CHAT_DOMAIN_THRESHOLD:
            q_lower = question.lower()
            if not any(kw in q_lower for kw in _DOMAIN_KEYWORDS):
                llm = get_llm(temperature=0.7)
                chat_history = self.memory.chat_memory.messages
                answer = _chat_directly(question, chat_history)
                self.memory.chat_memory.add_user_message(question)
                self.memory.chat_memory.add_ai_message(answer)
                if self.memory_window:
                    history = self.memory.chat_memory.messages
                    if len(history) > self.memory_window * 2:
                        trimmed = history[-(self.memory_window * 2):]
                        self.memory.chat_memory.messages = trimmed
                return answer, [], "chat"

        llm = get_llm()
        chat_history = self.memory.chat_memory.messages
        condensed = _condense_question(llm, question, chat_history)

        chain = self._get_chain()
        result = chain.invoke({"query": condensed})
        answer = result["result"]
        sources = _extract_sources(result.get("source_documents", []))

        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(answer)

        if self.memory_window:
            history = self.memory.chat_memory.messages
            if len(history) > self.memory_window * 2:
                trimmed = history[-(self.memory_window * 2):]
                self.memory.chat_memory.messages = trimmed

        return answer, sources, "rag"

    def reset(self):
        self.memory.clear()
        self._chain = None
        self._last_params = None


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

    k = 3
    prompt_mode = "anti_hallucination"
    search_type = "similarity"
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
        question = input("请输入问题: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not question:
            continue

        if question.startswith("/k "):
            try:
                k = int(question.split()[1])
                conv_mgr.k = k
                conv_mgr._chain = None
                print(f"检索数量已设置为 k={k}")
            except (IndexError, ValueError):
                print("用法: /k <数字>")
            continue

        if question.startswith("/mode "):
            mode = question.split(maxsplit=1)
            if len(mode) > 1 and mode[1] in PROMPT_TEMPLATES:
                prompt_mode = mode[1]
                conv_mgr.prompt_mode = prompt_mode
                conv_mgr._chain = None
                print(f"提示词模式已设置为: {prompt_mode}")
            else:
                print(f"可用模式: {', '.join(PROMPT_TEMPLATES.keys())}")
            continue

        if question.startswith("/search "):
            st = question.split(maxsplit=1)
            if len(st) > 1 and st[1] in ("similarity", "mmr"):
                search_type = st[1]
                conv_mgr.search_type = search_type
                conv_mgr._chain = None
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


if __name__ == "__main__":
    cli_qa()
