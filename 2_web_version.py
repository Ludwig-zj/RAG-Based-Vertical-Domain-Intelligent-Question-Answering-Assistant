import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. 网页界面设置 (UI) 与 CSS 注入
# ==========================================
st.set_page_config(page_title="企业智能问答系统", page_icon="🏢", layout="wide")

# 注入自定义 CSS 以实现 Chrome 极简/Material Design 风格
def inject_custom_css():
    st.markdown("""
    <style>
        /* 隐藏原生 Streamlit 痕迹 */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* 全局背景色与无衬线字体 */
        .stApp {
            background-color: #f8f9fa;
            font-family: 'Roboto', 'Segoe UI', 'Helvetica Neue', sans-serif;
        }
        
        /* 侧边栏现代化：白色背景、柔和阴影、去边框 */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            box-shadow: 2px 0 8px rgba(0,0,0,0.05);
            border-right: none;
        }
        
        /* 输入框卡片式阴影与圆角 */
        .stTextInput > div > div > input, .stSelectbox > div > div {
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.04);
            border: 1px solid #e0e0e0;
            padding: 10px 15px;
        }
        
        /* 按钮悬浮动画与质感 */
        .stButton > button {
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: all 0.2s ease-in-out;
            font-weight: 500;
        }
        .stButton > button:hover {
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }

        /* 聊天消息气泡样式优化 */
        [data-testid="stChatMessage"] {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            border: 1px solid #f1f3f4;
        }
        [data-testid="stChatMessage"] > div:first-child {
            /* 调整头像区域 */
            margin-right: 1rem;
        }
        
        /* 聊天输入框容器透明度调整 */
        .stChatInputContainer {
            background-color: transparent !important;
            padding-bottom: 20px;
        }
        
        /* Expander (折叠面板) 现代化 */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ==========================================
# 2. 会话状态管理 (Session State)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是企业智能助手。请先在左侧配置知识库，然后向我提问。"}]
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ==========================================
# 3. 动态知识库处理逻辑
# ==========================================
def process_uploaded_files(uploaded_files):
    """处理用户上传的文件，返回 Document 列表"""
    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            try:
                if file.name.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                elif file.name.lower().endswith((".txt", ".md")):
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                else:
                    continue
                # 给 document 标记真实的来源文件名
                for doc in docs:
                    doc.metadata['source'] = file.name 
                documents.extend(docs)
            except Exception as e:
                st.sidebar.error(f"❌ 读取 {file.name} 失败: {e}")
    return documents

def process_local_directory(dir_path):
    """处理本地文件夹绝对路径，返回 Document 列表"""
    documents = []
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        st.sidebar.error("❌ 文件夹路径无效或不存在。")
        return documents
        
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file.lower().endswith((".txt", ".md")):
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
            except Exception as e:
                st.sidebar.warning(f"⚠️ 跳过文件 {file} (读取失败): {e}")
    return documents

def build_vector_db(documents):
    """文本切分并构建向量数据库"""
    if not documents:
        return None
    
    # 文本切分 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    
    # 构建 FAISS 向量库 (缓存在 session_state 中)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(split_docs, embeddings)
    return vector_db

# ==========================================
# 4. 侧边栏：系统配置与知识库挂载
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ 系统配置")
    api_key = st.text_input("Gemini API Key", type="password", help="以 AIza 开头的密钥")
    
    st.markdown("### 🗂️ 知识库挂载")
    kb_mode = st.radio("选择数据源", ["上传文件", "本地文件夹路径"])
    
    documents_to_process = []
    
    if kb_mode == "上传文件":
        uploaded_files = st.file_uploader(
            "支持批量上传 .txt, .md, .pdf", 
            type=['txt', 'md', 'pdf'], 
            accept_multiple_files=True
        )
        if st.button("构建知识库 (上传)", use_container_width=True) and uploaded_files:
            with st.spinner("正在解析上传的文件..."):
                documents_to_process = process_uploaded_files(uploaded_files)
                
    else:
        folder_path = st.text_input("输入本地文件夹绝对路径", placeholder="例如: D:\\my_docs\\")
        if st.button("构建知识库 (路径)", use_container_width=True) and folder_path:
            with st.spinner("正在扫描并解析本地文件夹..."):
                documents_to_process = process_local_directory(folder_path)

    # 统一构建流程
    if documents_to_process:
        with st.spinner("正在切分文本并构建向量索引..."):
            st.session_state.vector_db = build_vector_db(documents_to_process)
            if st.session_state.vector_db:
                st.success("✅ 知识库构建完成！您可以开始提问了。")
            else:
                st.error("❌ 未能从数据源提取到有效文本。")

    if st.session_state.vector_db:
        st.info("💡 当前已加载知识库。")
    
    if st.button("🧹 清空会话历史", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "会话已清空。您可以继续提问。"}]
        st.rerun()

# ==========================================
# 5. 主页面：聊天交互流 (Chat UI)
# ==========================================
st.title("企业智能助理 🕵️‍♂️")

# 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果历史消息中带有溯源信息，则用优雅的 Markdown 引用展示
        if "source_docs" in msg and msg["source_docs"]:
            with st.expander("📚 查看参考来源"):
                for idx, doc in enumerate(msg["source_docs"]):
                    source_name = doc.metadata.get('source', '未知文件')
                    st.markdown(f"> **[{idx+1}] {source_name}**\n>\n> *...{doc.page_content[:150].replace(chr(10), ' ')}...*")

# 聊天输入框
if prompt := st.chat_input("询问你的企业知识库... (例如: 公司WiFi密码是什么?)"):
    
    # 1. 拦截检查
    if not api_key:
        st.warning("⚠️ 请先在左侧边栏输入您的 Gemini API Key！")
        st.stop()
    
    if not st.session_state.vector_db:
        st.warning("⚠️ 知识库尚未构建！请先在左侧上传文件或输入文件夹路径进行构建。")
        st.stop()
        
    # 2. 将用户问题追加到历史并展示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. 生成 AI 回答
    with st.chat_message("assistant"):
        with st.spinner("正在企业知识库中检索并思考..."):
            try:
                # 配置环境变量
                os.environ["GOOGLE_API_KEY"] = api_key
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
                
                # 执行向量检索 (RAG)
                docs = st.session_state.vector_db.similarity_search(prompt, k=3)
                
                # 组装上下文信息
                context_texts = []
                for idx, doc in enumerate(docs):
                    source = doc.metadata.get('source', 'Unknown')
                    content = doc.page_content
                    context_texts.append(f"--- 资料 [{idx+1}] (来源: {source}) ---\n{content}")
                context_str = "\n\n".join(context_texts)
                
                # 构建 Prompt
                rag_prompt = f"""
                你是一个专业、现代化的企业内部 AI 助手。
                请**严格根据**以下提供的【参考资料】来回答用户的【问题】。
                如果参考资料中没有相关答案，请直接回答“对不起，当前知识库中未找到相关信息”。
                回答请尽量使用 Markdown 格式，保持排版优雅清晰。

                【参考资料】：
                {context_str}

                【问题】：
                {prompt}
                """
                
                # 调用大模型
                response = llm.invoke(rag_prompt)
                answer = response.content
                
                # 展示回答
                st.markdown(answer)
                
                # 优雅的溯源展示
                with st.expander("📚 查看参考来源"):
                    for idx, doc in enumerate(docs):
                        source_name = doc.metadata.get('source', '未知文件')
                        st.markdown(f"> **[{idx+1}] {source_name}**\n>\n> *...{doc.page_content[:150].replace(chr(10), ' ')}...*")
                
                # 保存到 Session State
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "source_docs": docs # 存储检索到的原始文档以便渲染历史记录
                })
                
            except Exception as e:
                st.error(f"❌ 系统发生错误: {str(e)}")