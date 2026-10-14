import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

DB_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="rag_evaluation_512")

print("Loading Embedding Model (BAAI/bge-m3)...")
embedding_model = SentenceTransformer("BAAI/bge-m3")

def calculate_metrics(eval_file="evaluation_set/eval_question.json"):
    if not os.path.exists(eval_file):
        print(f"Error : No eval file found at {eval_file}")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        question_data = json.load(f)

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    mrr_sum = 0
    total = len(question_data)

    results_detail = []
    
    for item in question_data:
        q_id = item["id"]
        question = item["question"]
        expected_ids = item["expected_chunk_ids"]

        q_embedding = embedding_model.encode([question]).tolist()

        result = collection.query(
            query_embeddings=q_embedding,
            n_results=5
        )

        retrieved_ids = result["ids"][0] if result["ids"] else []

        # คำนวณ Hit@k
        found_in_top1 = any(e_id in retrieved_ids[:1] for e_id in expected_ids)
        found_in_top3 = any(e_id in retrieved_ids[:3] for e_id in expected_ids)
        found_in_top5 = any(e_id in retrieved_ids[:5] for e_id in expected_ids)

        if found_in_top1: hit_at_1 += 1
        if found_in_top3: hit_at_3 += 1
        if found_in_top5: hit_at_5 += 1

        # คำนวน Mean Reciprocal Rank (MRR)
        rank = 0
        for idx, retrieved_id in enumerate(retrieved_ids):
            if retrieved_id in expected_ids:
                rank = idx + 1
                break
        reciprocal_rank = (1.0 / rank) if rank > 0 else 0.0
        mrr_sum += reciprocal_rank

        results_detail.append({
           "id": q_id,
            "question": question,
            "expected_chunk_ids": expected_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "rank": rank,
            "reciprocal_rank": reciprocal_rank
        })


    # คำนวนค่าเฉลี่ยเป็น %
    avg_hit1 = (hit_at_1 / total) * 100
    avg_hit3 = (hit_at_3 / total) * 100
    avg_hit5 = (hit_at_5 / total) * 100
    avg_mrr = mrr_sum / total

    print("\n=================== 📊 RESULT RETRIEVAL EVALUATION 📊 ===================")
    print(f"จำนวนคำถามทั้งหมด: {total} ข้อ")
    print(f"Hit@1 : {avg_hit1:.2f}% ({hit_at_1}/{total})")
    print(f"Hit@3 : {avg_hit3:.2f}% ({hit_at_3}/{total})")
    print(f"Hit@5 : {avg_hit5:.2f}% ({hit_at_5}/{total})")
    print(f"MRR   : {avg_mrr:.4f}")
    print("========================================================================\n")

    os.makedirs("outputs", exist_ok=True)
    summary_data = {
        "config": "bge-m3_chunk512",
        "total_questions": total,
        "hit_at_1": avg_hit1,
        "hit_at_3": avg_hit3,
        "hit_at_5": avg_hit5,
        "mrr": avg_mrr
    }

    with open("outputs/retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary_data, "details": results_detail}, f, ensure_ascii=False, indent=2)
        
    print(f"Detail results saved to {os.getcwd()}/outputs/retrieval_metrics.json")

if __name__ == "__main__":
    calculate_metrics()
        
    