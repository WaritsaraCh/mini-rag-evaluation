import json
import os
import chromadb

DB_PATH = "./chroma_db"
EVAL_PATH = "evaluation_set/eval_question.json"

def inspect_chunks_and_create_template():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name="rag_evaluation_512")

    data = collection.get()
    documents = data['documents']
    metadatas = data['metadatas']
    ids = data['ids']

    print(f"Total Documents: {len(documents)}")

    print(f"--- Sample data chunk ---")
    for i in range(min(5, len(documents))):
        print(f"ID: {ids[i]}")
        print(f"Source: {metadatas[i]['source']} (Page {metadatas[i]['page']})")
        print(f"Content: {documents[i][:150]}...")
        print("-" * 50)

    os.makedirs("evaluation_set", exist_ok=True)
    if not os.path.exists(EVAL_PATH):
        sample_eval_data = [
            {
                "id": f"q{i+1}",
                "question": f"ตัวอย่างคำถามข้อที่ {i+1} (กรุณารับแก้ไขคำถามจริง)",
                "expected_chunk_ids": [ids[i] if i < len(ids) else "chunk_id_here"],
                "reference_answer": "ตัวอย่างคำตอบอ้างอิงสั้นๆ"
            }
            for i in range(30)
        ]
        with open(EVAL_PATH, "w", encoding="utf-8") as f:
            json.dump(sample_eval_data, f, ensure_ascii=False, indent=4)
        print(f"Created sample eval file: {EVAL_PATH}")

    else:
        print(f"Eval file already exists at {EVAL_PATH}")


if __name__ == "__main__":
    inspect_chunks_and_create_template()
    