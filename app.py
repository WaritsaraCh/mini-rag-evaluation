import os
import json
import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(
    page_title="Mini RAG Evaluation Project",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_vector_db():
    DB_PATH = "./chroma_db"
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="rag_evaluation_512")
    model = SentenceTransformer("BAAI/bge-m3")
    
    
    if collection.count() == 0:
        if os.path.exists("src/ingest.py"):
            st.info("⏳ ไม่พบ Vector Store บน Cloud ระบบกำลังประมวลผล Ingestion อัตโนมัติ...")
            from src.ingest import process_documents
            process_documents()
            st.success("✅ สร้าง Vector Store สำเร็จแล้ว!")
            collection = client.get_collection(name="rag_evaluation_512")
            
    return collection, model

@st.cache_resource
def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    return None
collection, embedding_model = load_vector_db()
gemini_model = init_gemini()

st.title("🔍 Mini RAG Evaluation Dashboard")
st.caption("ระบบทดสอบและประเมินผลประสิทธิภาพของ RAG Pipeline บนเอกสารสุขภาพ (3อ.2ส.)")

tab1, tab2 = st.tabs(["🤖 Live RAG Demo", "📊 Evaluation Results"])


with tab1:
    st.header("ทดลองใช้งาน RAG Interactive Demo")
    st.write("พิมพ์คำถามเพื่อทดสอบการดึงข้อมูลจาก Vector Store (Chroma DB) และการสร้างคำตอบด้วย Gemini")

    user_query = st.text_input("ใส่คำถามของคุณที่นี่:", placeholder="เช่น เกณฑ์การวินิจฉัยโรคความดันโลหิตสูงดูจากค่าใด?")
    top_k = st.slider("จำนวน Chunks ที่ต้องการดึง (Top K):", min_value=1, max_value=5, value=3)

    if st.button("ค้นหาและตอบคำถาม", type="primary"):
        if not user_query.strip():
            st.warning("กรุณากรอกคำถามก่อนครับ")
        else:
            with st.spinner("กำลังค้นหาข้อมูลและสร้างคำตอบ..."):
                # 1. Retrieval
                q_embed = embedding_model.encode([user_query]).tolist()
                res = collection.query(query_embeddings=q_embed, n_results=top_k)
                
                retrieved_docs = res["documents"][0] if res["documents"] else []
                retrieved_metas = res["metadatas"][0] if res["metadatas"] else []

                context_text = "\n\n---\n\n".join(retrieved_docs)

                # 2. Generation
                if gemini_model:
                    prompt = f"""คุณคือผู้ช่วยตอบคำถาม ให้ตอบคำถามโดยอ้างอิงจากบริบทที่กำหนดให้เท่านั้น หากในบริบทไม่มีข้อมูล ให้ตอบว่า 'ไม่พบข้อมูลในเอกสาร'

                    [บริบท]:
                    {context_text}

                    [คำถาม]:
                    {user_query}
                    """
                    answer = gemini_model.generate_content(prompt).text
                else:
                    answer = "⚠️ กรุณาตั้งค่า GEMINI_API_KEY ในไฟล์ .env เพื่อเปิดใช้งาน LLM Answer Generation"

            # Display Results
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🤖 คำตอบจาก RAG System")
                st.info(answer)

            with col2:
                st.subheader("📚 Chunks ที่ดึงมาอ้างอิง (Retrieved Context)")
                for idx, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
                    with st.expander(f"Chunk #{idx+1} | Source: {meta.get('source')} (Page {meta.get('page')})"):
                        st.write(doc)


with tab2:
    st.header("ผลการประเมินประสิทธิภาพ (RAG Metrics)")
    
    metrics_path = "outputs/retrieval_metrics.json"
    llm_eval_path = "outputs/llm_eval_results.json"

 
    st.subheader("1. Retrieval Performance Metrics (Hit@k & MRR)")
    
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        
        summary = metrics_data.get("summary", {})
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Hit@1", f"{summary.get('hit_at_1', 0):.2f}%")
        m_col2.metric("Hit@3", f"{summary.get('hit_at_3', 0):.2f}%")
        m_col3.metric("Hit@5", f"{summary.get('hit_at_5', 0):.2f}%")
        m_col4.metric("MRR (Mean Reciprocal Rank)", f"{summary.get('mrr', 0):.4f}")
    else:
        st.warning("⚠️ ไม่พบไฟล์ผลลัพธ์ `outputs/retrieval_metrics.json` กรุณารัน `python src/evaluate_retrieval.py` ก่อนครับ")

    st.markdown("---")

    st.subheader("2. เปรียบเทียบคุณภาพคำตอบ: No-RAG vs With RAG")
    
    if os.path.exists(llm_eval_path):
        with open(llm_eval_path, "r", encoding="utf-8") as f:
            llm_results = json.load(f)
        
        if isinstance(llm_results, list) and len(llm_results) > 0:
            df_llm = pd.DataFrame(llm_results)
            
            required_cols = ["id", "question", "reference_answer", "answer_no_rag", "answer_with_rag"]
            available_cols = [col for col in required_cols if col in df_llm.columns]
            
            st.dataframe(df_llm[available_cols], use_container_width=True)
            
            if "id" in df_llm.columns:
                st.subheader("🔍 ตรวจสอบคำตอบทีละข้อ")
                selected_id = st.selectbox("เลือกข้อที่ต้องการดูรายละเอียด:", df_llm["id"].tolist())
                selected_item = df_llm[df_llm["id"] == selected_id].iloc[0]

                q_col1, q_col2 = st.columns(2)
                with q_col1:
                    st.markdown(f"**คำถาม:** {selected_item.get('question', '')}")
                    st.markdown(f"**คำตอบอ้างอิง (Ground Truth):** {selected_item.get('reference_answer', '')}")
                    st.error(f"**❌ No-RAG Answer:**\n{selected_item.get('answer_no_rag', '')}")
                    
                with q_col2:
                    st.success(f"**✅ With RAG Answer:**\n{selected_item.get('answer_with_rag', '')}")
                    with st.expander("ดู Context ที่ส่งให้ LLM"):
                        st.write(selected_item.get("context_used", ""))
        else:
            st.warning("⚠️ ไฟล์ `outputs/llm_eval_results.json` ยังไม่มีข้อมูล กรุณารัน `python src/evaluate_llm.py` ก่อนครับ")
    else:
        st.warning("⚠️ ไม่พบไฟล์ผลลัพธ์ `outputs/llm_eval_results.json` กรุณารัน `python src/evaluate_llm.py` ก่อนครับ")