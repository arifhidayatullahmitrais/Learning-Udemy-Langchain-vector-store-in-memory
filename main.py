import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

if __name__ == '__main__':
    print(os.environ["OPENAI_API_KEY"])
    pdf_path = "/Users/arif_h667/Learning/Udemy/Langchain/vector-store-in-memory/example.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    document = loader.load()
    print(document)