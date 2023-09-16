import gradio as gr
from langchain.text_splitter import CharacterTextSplitter


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

import os

os.environ["OPENAI_API_KEY"] = 'xxxxxxxxx'

with open("tv_data.txt",encoding="UTF-8") as f:
    tv_data = f.read()

text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 80,
    chunk_overlap  = 0,
    length_function = len,
    is_separator_regex = True,
)

docs = text_splitter.create_documents([tv_data])
db = FAISS.from_documents(docs, OpenAIEmbeddings())
db.save_local("tv_sale_data")


def initialize_sales_bot(vector_store_dir: str = "tv_sale_data"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    global SALES_BOT
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                            retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                      search_kwargs={"score_threshold": 0.7}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = False

    ans = SALES_BOT({"query": message})
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    else:
        return "很抱歉，没有相关数据"


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="电视机销售平台",
        chatbot=gr.Chatbot(height=500),
    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    initialize_sales_bot()
    launch_gradio()
