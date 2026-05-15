# Streamlit 快速开发指南

## 什么是 Streamlit？

Streamlit 是一个开源的 Python 框架，专为数据科学家和机器学习工程师设计，可以在几分钟内将 Python 脚本转换为交互式 Web 应用。无需前端开发经验，只需 Python 知识即可构建美观的数据应用。

## 核心特性

### 1. 极简开发模式

Streamlit 采用声明式编程范式，开发者只需描述"想要什么"，而不需要关心"如何实现"。一个最简单的 Streamlit 应用只需要几行代码：

```python
import streamlit as st
st.title("Hello World")
st.write("这是一个 Streamlit 应用")
```

运行命令 `streamlit run app.py` 即可在浏览器中查看应用。

### 2. 丰富的组件库

Streamlit 提供了大量内置组件，覆盖数据展示、用户交互、图表可视化等场景：

- **文本展示**：`st.markdown`、`st.code`、`st.latex`、`st.json`
- **数据展示**：`st.dataframe`、`st.table`、`st.metric`
- **图表**：`st.line_chart`、`st.bar_chart`、`st.pyplot`、`st.plotly_chart`
- **用户输入**：`st.text_input`、`st.slider`、`st.selectbox`、`st.checkbox`
- **媒体**：`st.image`、`st.audio`、`st.video`
- **布局**：`st.sidebar`、`st.columns`、`st.tabs`、`st.expander`
- **进度**：`st.progress`、`st.spinner`、`st.balloons`

### 3. 会话状态管理

`st.session_state` 是 Streamlit 的状态管理机制，用于在应用重新运行时保持数据。每次用户与组件交互，Streamlit 会从头到尾重新执行脚本，`session_state` 确保关键数据不会丢失。

```python
if "count" not in st.session_state:
    st.session_state.count = 0
st.session_state.count += 1
st.write(f"点击次数: {st.session_state.count}")
```

### 4. 缓存机制

`@st.cache_data` 和 `@st.cache_resource` 装饰器可以缓存函数返回值，避免重复计算：

- `@st.cache_data`：缓存数据（如 DataFrame、列表、字典）
- `@st.cache_resource`：缓存资源（如数据库连接、模型实例）

```python
@st.cache_data(ttl=300)
def load_data():
    return pd.read_csv("data.csv")
```

## 在 RAG 系统中的应用

Streamlit 非常适合作为 RAG（检索增强生成）系统的前端界面：

1. **对话界面**：`st.chat_message` 和 `st.chat_input` 提供类似 ChatGPT 的对话体验
2. **参数调节**：侧边栏滑块和选择框实时调整检索参数（k值、阈值、检索策略）
3. **来源展示**：`st.expander` 折叠展示检索来源文档
4. **反馈收集**：按钮组件收集用户对回答质量的评价
5. **数据可视化**：展示向量数据库统计信息和反馈分析

## 部署方式

- **本地运行**：`streamlit run app.py`
- **Streamlit Community Cloud**：免费部署，直接关联 GitHub 仓库
- **Docker**：官方提供基础镜像，适合企业级部署
- **云服务器**：配合 Nginx 反向代理部署到 VPS

## 最佳实践

1. 将耗时操作放在缓存函数中，避免每次交互重新计算
2. 使用 `st.session_state` 管理对话历史和用户状态
3. 合理使用布局组件（columns、tabs、expander）组织界面
4. 大型应用建议使用多页面（multipage app）结构
5. 敏感信息（API Key）通过环境变量或 `.env` 文件管理
