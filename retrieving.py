import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import SecretStr

load_dotenv()

if __name__ == '__main__':
    print("loading vector store...")
    embeddings = OpenAIEmbeddings(
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        model=os.environ["OPENAI_EMBEDDING_MODEL"],
    )

    load_vector_store = FAISS.load_local(
        "faiss_index_react",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        ChatOpenAI(
            api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
            model=os.environ["OPENAI_MODEL"],
        ), prompt
    )
    retrieval_chain = create_retrieval_chain(
        load_vector_store.as_retriever(), combine_docs_chain
    )
    response = retrieval_chain.invoke({
        "input": "what is the phone number?"
    })
    print(response["answer"])