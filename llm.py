import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylibs"))

from config import PROJECT_ROOT

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

CHAT_SYSTEM_PROMPT = """你是一个智能助手，基于 DeepSeek 大模型驱动。

【重要规则】
- 当用户问的问题**不在**知识库文档范围内（如学习路线推荐、生活常识、编程建议、开放性问题等），请直接用你的通用知识回答，不要说"无法回答"或"知识库中没有"，就像一个正常的 AI 助手一样帮助用户。
- 当用户问的是**文档相关**的具体技术问题，系统会自动走 RAG 检索流程，你不需要特别说明。
- 日常问候、闲聊、翻译、写作等任何问题都请正常回答。

回答风格：简洁、专业、有帮助。中文回答。"""


def get_llm(temperature=0):
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    model_name = os.getenv("MODEL_NAME", "deepseek-chat")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 未配置，请在 .env 文件中设置")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL 未配置，请在 .env 文件中设置")
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=base_url,
    )


def chat_directly(question, chat_history_messages=None):
    llm = get_llm(temperature=0.7)
    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)]
    if chat_history_messages:
        messages.extend(chat_history_messages[-10:])
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content


def condense_question(llm, question, chat_history):
    if not chat_history:
        return question
    history_text = ""
    for msg in chat_history[-10:]:
        content = str(msg.content).replace("{", "{{").replace("}", "}}")
        if isinstance(msg, HumanMessage):
            history_text += f"用户: {content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"助手: {content}\n"
    if not history_text.strip():
        return question
    safe_q = question.replace("{", "{{").replace("}", "}}")
    condense_prompt = f"""根据以下对话历史和最新问题，将问题重写为一个独立的、完整的问题。

对话历史：
{history_text}

最新问题：{safe_q}

独立问题："""
    try:
        response = llm.invoke([HumanMessage(content=condense_prompt)])
        condensed = response.content.strip()
        return condensed if condensed else question
    except Exception:
        return question


def verify_api():
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    model_name = os.getenv("MODEL_NAME", "deepseek-chat")

    print("-" * 50)
    print("  环境配置与API验证")
    print("-" * 50)

    if not api_key:
        print("  ❌ OPENAI_API_KEY 未设置，请检查 .env 文件")
        return False
    if not base_url:
        print("  ❌ OPENAI_BASE_URL 未设置，请检查 .env 文件")
        return False

    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"  ✅ API Key: {masked_key}")
    print(f"  ✅ Base URL: {base_url}")
    print(f"  ✅ Model: {model_name}")

    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=api_key,
            openai_api_base=base_url,
            max_tokens=10,
        )
        response = llm.invoke([HumanMessage(content="说'OK'")])
        print(f"  ✅ API 连接成功！模型响应: {response.content.strip()}")
        return True
    except Exception as e:
        error_str = str(e)
        print(f"  ❌ API 连接失败: {error_str}")
        if "401" in error_str or "Unauthorized" in error_str:
            print("     → 原因: API Key 无效或已过期")
        elif "402" in error_str or "Insufficient" in error_str:
            print("     → 原因: 账户余额不足，请充值")
        elif "404" in error_str or "Not Found" in error_str:
            print("     → 原因: BASE_URL 路径错误，检查是否包含 /v1 后缀")
        elif "Connection" in error_str or "timeout" in error_str.lower():
            print("     → 原因: 网络连接失败或服务不可达")
        return False
