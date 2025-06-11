from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os


path1 = r"D:\tools\new\llmgogogo\health\vectorstore_code"
path2 = r"D:\tools\new\llmgogogo\health\vectorstore_example"
file_paths1 = [
    r"D:\tools\new\llmgogogo\health\生命质量数据字段说明.csv",
r"D:\tools\new\llmgogogo\health\体质测试数据字段说明.csv",
r"D:\tools\new\llmgogogo\health\营养与膳食情况数据字段说明.csv"
]
file_paths2 = [
    r"D:\tools\new\llmgogogo\health\体质测试_example.csv",
    r"D:\tools\new\llmgogogo\health\营养与膳食_example.csv",
    r"D:\tools\new\llmgogogo\health\生命质量_example.csv",
]
def get_retriever_code(file_path,vectordb):
    all_docs1 = []
    for path in file_path:
        loader = CSVLoader(file_path=path, encoding='utf-8')
        data = loader.load()
        all_docs1.extend(data)
    if os.path.exists(vectordb):
        vectordb = Chroma(persist_directory=vectordb, embedding_function=embedding)
    else:
        vectordb = Chroma.from_documents(documents=all_docs1,
            embedding=embedding,
            persist_directory=vectordb
        )
    retriever = vectordb.as_retriever()
    return retriever

def get_retriever_example(file_path,vectordb):
    all_docs2 = []
    for path in file_path:
        loader = CSVLoader(file_path=path, encoding='utf-8')
        data = loader.load()
        all_docs2.extend(data)
    if os.path.exists(vectordb):
        vectordb = Chroma(persist_directory=vectordb, embedding_function=embedding)
    else:
        vectordb = Chroma.from_documents(documents=all_docs2,
            embedding=embedding,
            persist_directory=vectordb
        )
    retriever = vectordb.as_retriever()
    return retriever

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh")

def get_health_retrievers():
    retriever_code = get_retriever_code(file_paths1, path1)
    retriever_example = get_retriever_example(file_paths2, path2)
    return retriever_code, retriever_example