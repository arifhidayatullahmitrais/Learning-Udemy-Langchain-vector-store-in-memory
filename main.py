import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pydantic import SecretStr

load_dotenv()

if __name__ == '__main__':
    print(os.environ["OPENAI_API_KEY"])
    pdf_path = "/Users/arif_h667/Learning/Udemy/Langchain/vector-store-in-memory/example.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    print(documents)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=25, separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    print(docs)

    embeddings = OpenAIEmbeddings(
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        model=os.environ["OPENAI_EMBEDDING_MODEL"],
    )

    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_index_react")