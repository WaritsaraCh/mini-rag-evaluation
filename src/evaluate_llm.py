import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="rag_evaluation_512")
embedding_model = SentenceTransformer("BAAI/bge-m3")

def get_retrivaled_context(question, top_k=5):
    q_embed = embedding_model.encode([question]).tolist()
    res =collection.query(query_embeddings=q_embed, n_results=top_k)
    docs = res['documents'][0] if res["documents"][0] else []
    return "\n---\n".join(docs)

def generate_answer_no_rag(question):
    prompt = f"กรุณาตอบคำถามต่อไปนี้ตามความรู้ของคุณอย่างสั้นและตรงประเด็น:\nคำถาม: {question}"

    try:
        respone = model.generate_content(prompt)
        return respone.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def generate_answer_rag(question, context):
    prompt = f"""คุณคือผู้ช่วยตอบคำถาม ให้ตอบคำถามโดยอ้างอิงจากบริบทที่กำหนดให้เท่านั้น หากในบริบทไม่มีข้อมูล ให้ตอบว่า 'ไม่พบข้อมูลในเอกสาร'

    [บริบท]:
    {context}

    [คำถาม]:
    {question}
    """

    try:
        respone = model.generate_content(prompt)
        return respone.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def run_ab_testing(eval_file="evaluation_set/eval_question.json", sample_size=10):
    if not os.path.exists(eval_file):
        print(f"Error: No eval file found at {eval_file}")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        question = json.load(f)

    eval_result = []
    for item in question[:sample_size]:
        q_id = item["id"]
        question = item["question"]
        reference_answer = item["reference_answer"]

        print(f"\nEvaluating {q_id}: {question[:40]}...")

        ans_no_rag = generate_answer_no_rag(question)

        context = get_retrivaled_context(question)
        ans_rag = generate_answer_rag(question, context)

        eval_result.append({
            "id":q_id,
            "question":question,
            "reference_answer":reference_answer,
            "answer_no_rag":ans_no_rag,
            "answer_rag":ans_rag,
            "context_used":context
        })
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/llm_eval_results.json"
    with open(out_path,'w', encoding='utf-8') as f:
        json.dump({"eval_results":eval_result},f,ensure_ascii=False, indent=2)

    print("=============================================")
        
if __name__ == "__main__":
    run_ab_testing()
        
        
