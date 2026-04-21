import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# 1. 网页界面设置 (UI)
# ==========================================
st.set_page_config(page_title="企业智能助手", page_icon="🏢")
st.title("企业内部机密问答助手 🕵️‍♂️")
st.markdown("基于 **Gemini 2.5 大模型** + **本地向量检索 (RAG)** 技术构建")

# 在侧边栏设置密码输入框 (商业产品的标准做法)
st.sidebar.header("⚙️ 系统配置")
api_key = st.sidebar.text_input("请输入 Gemini API Key", type="password", help="以 AIza 开头的密钥")

# ==========================================
# 2. 核心逻辑：加载并缓存知识库
# ==========================================
# @st.cache_resource 是 Streamlit 的魔法装饰器。
# 它保证了这个耗时的加载过程在网页启动时只运行一次，后面直接用缓存。
@st.cache_resource
def load_knowledge_base():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    my_knowledge = [
        Document(page_content="公司的WiFi密码是：AI_Love_888。"),
        Document(page_content="公司的新员工入职指南：第一天请去前台找行政总监 Lisa 领取电脑。"),
        Document(page_content="本公司的核心价值观是：绝不内卷，每天下午六点准时下班。")
    ]
    return FAISS.from_documents(my_knowledge, embeddings)

vector_db = load_knowledge_base()

# ==========================================
# 3. 用户交互与问答 (相当于之前的 print 和 input)
# ==========================================
# 生成一个输入框供用户提问
question = st.text_input("请问有什么我可以帮您？", placeholder="例如：新人第一天上班该干嘛？")

# 生成一个发送按钮，当被点击时执行下面的代码
if st.button("开始查询"):
    # 基础的错误拦截
    if not api_key:
        st.error("⚠️ 请先在左侧边栏输入您的 API Key！")
    elif not question:
        st.warning("⚠️ 请输入您的问题！")
    else:
        # st.spinner 会在网页上转圈圈，告诉用户“AI 正在思考”
        with st.spinner("正在企业知识库中检索并思考..."):
            
            # 1. 配置临时密码
            os.environ["GOOGLE_API_KEY"] = api_key
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            
            # 2. 检索资料
            docs = vector_db.similarity_search(question, k=1)
            retrieved_info = docs[0].page_content
            
            # 3. 组装 Prompt
            rag_prompt = f"""
            你是一个专业的企业内部 AI 助手。
            请**严格根据**以下提供的【参考资料】来回答用户的【问题】。
            如果参考资料中没有相关答案，请直接回答“对不起，内部资料中未找到相关信息”。

            【参考资料】：
            {retrieved_info}

            【问题】：
            {question}
            """
            
            # 4. 获取大模型回答
            try:
                response = llm.invoke(rag_prompt)
                
                # 5. 在网页上漂亮地展示结果
                st.success("✅ 查询成功！")
                st.info(response.content) # 蓝色高亮框展示回答
                
                # 折叠面板，用来展示底层检索到了什么内容（增强透明度）
                with st.expander("点击查看 AI 参考的底层机密资料"):
                    st.write(retrieved_info)
                    
            except Exception as e:
                st.error(f"❌ 调用大模型时发生错误：{e}")