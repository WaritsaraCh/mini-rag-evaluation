import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

DB_PATH = './chroma_db'
client = chromadb.PersistentClient(path=DB_PATH)

print('Loading Embedding Model (BAAI/bge-m3)...')
embedding_model = SentenceTransformer('BAAI/bge-m3')

collection = client.get_or_create_collection(name='rag_evaluation_512')


def parse_pdf_with_llamaparse(pdf_path):
    
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key and hasattr(st, "secrets") and "LLAMA_CLOUD_API_KEY" in st.secrets:
        api_key = st.secrets["LLAMA_CLOUD_API_KEY"]
        
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=True
    )

    documents = parser.load_data(pdf_path)

    pages_data = []
    for idx, doc in enumerate(documents):
        if doc.text.strip():
            pages_data.append({
                "page_number": idx +1,
                "text": doc.text
            })

    return pages_data


def process_documents(data_dir="data"):

    pdf_file = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_file:
        print(f"Error :No PDF found in {data_dir}")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 50,
        separators = ['\n\n', '\n', '.', ' ']
    )

    documents = []
    metadatas = []
    ids = []

    for file_path in pdf_file:
        filename = os.path.basename(file_path)
        print(f"Processing file: {filename}")
        pages = parse_pdf_with_llamaparse(file_path)

        for page in pages:
            chunks = text_splitter.split_text(page['text'])
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{filename}_page_{page['page_number']}_chunk_{idx + 1}"

                documents.append(chunk)
                metadatas.append({
                    "source": filename,
                    "page": page['page_number'],
                    "chunk_number": idx + 1
                })
                ids.append(chunk_id)

    if documents:
        print("Generating Embeddings...")
        embeddings = embedding_model.encode(
            documents,
            batch_size = 32,
            show_progress_bar = True,
            convert_to_numpy=True
        ).tolist()

        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully added {len(documents)} chunks to ChromaDB")

if __name__ == "__main__":
    process_documents()

    



    
