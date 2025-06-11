from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader,TextLoader,UnstructuredExcelLoader,UnstructuredWordDocumentLoader
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from agent_tools import tools,llm,prompt
from langchain.agents import AgentExecutor
import streamlit as st
import os
import tempfile
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent

st.title('Health Agent')
uploaded_files = st.sidebar.file_uploader(
    label="文件上传", type=["txt",'pdf','word','xlsx'], accept_multiple_files=True
)
@st.cache_resource()
def load_files(uploaded_files):
    docs=[]
    temp_dir = tempfile.TemporaryDirectory(dir=r'D:\\')
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        if  file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_filepath)
            docs.extend(loader.load_and_split())

        elif file.name.endswith('.txt'):
            loader = TextLoader(temp_filepath, encoding="utf-8")
            docs.extend(loader.load_and_split())
        elif file.name.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(temp_filepath)
            docs.extend(loader.load_and_split())
        elif file.name.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(temp_filepath)
            docs.extend(loader.load_and_split())
    print(docs)
    if not docs:
        st.warning("未能从文件中加载任何文档，请检查格式或内容。")
        return None

    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh")
    vectordb = Chroma.from_documents(documents=docs,embedding=embedding,)
    retriever = vectordb.as_retriever()
    return retriever


retriever = None
tool1 = None

if uploaded_files:
    retriever = load_files(uploaded_files)
    if retriever:
        tool1 = create_retriever_tool(
            retriever,
            description="根据问题，查询上传的文档或文件并进行回复",
            name="search_document",
        )
        tools.append(tool1)
    else:
        st.error("文档加载失败，无法创建检索器。")


if tool1:
    st.success("文档已上传并建立索引，可以开始提问。")
# else:
#     st.info("请上传文档以启用文档检索功能。")


print("当前可用工具：", [tool.name for tool in tools])

if "messages" not in st.session_state or st.sidebar.button("清空聊天记录"):
    st.session_state["messages"] = [{"role": "assistant", "content": "您好，我是AI助手，我可以查询文档"}]

# 加载历史聊天记录
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

msg = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history",chat_memory=msg,return_messages=True,output_key = 'output')
agent = create_react_agent(llm,tools,prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

user_query = st.chat_input(placeholder="请输入问题。")

if user_query:

    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        config = {"callbacks": [st_cb]}
        response = agent_executor.invoke({"input": user_query}, config=config)
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        st.write(response["output"])

