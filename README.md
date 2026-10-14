# 🔍 Mini RAG Evaluation Project

โครงการตัวอย่างสำหรับสร้างและประเมินระบบ Retrieval-Augmented Generation (RAG) จากเอกสาร PDF ภาษาไทย โดยระบบจะแปลงเนื้อหาในเอกสารเป็นเวกเตอร์ จัดเก็บใน ChromaDB ค้นหาข้อความที่เกี่ยวข้องกับคำถาม และส่งบริบทที่ค้นพบให้ Google Gemini สร้างคำตอบ

โครงการนี้มีทั้งสคริปต์สำหรับประมวลผลข้อมูลและ Streamlit Dashboard สำหรับทดลองถามคำถาม รวมถึงดูผลการประเมินระบบ

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mini-rag-evaluation.streamlit.app)

> 🔗 **Live Demo:** [https://mini-rag-evaluation.streamlit.app](https://mini-rag-evaluation.streamlit.app/)
<!-- > 📁 **GitHub Repo:** [https://github.com/USERNAME/mini-rag-evaluation](https://github.com/WaritsaraCh/mini-rag-evaluation) -->


## ความสามารถหลัก

- อ่านและแปลงเอกสาร PDF เป็น Markdown ด้วย LlamaParse
- แบ่งข้อความเป็นส่วนย่อยขนาด 512 ตัวอักษร โดยซ้อนทับกัน 50 ตัวอักษร
- สร้างเวกเตอร์แทนข้อความด้วยโมเดล `BAAI/bge-m3`
- จัดเก็บเวกเตอร์และข้อมูลกำกับใน ChromaDB
- ค้นหาข้อความที่เกี่ยวข้องแบบ Top K
- สร้างคำตอบจากบริบทที่ค้นพบด้วย Google Gemini
- ประเมินประสิทธิภาพการค้นคืนด้วย Hit@1, Hit@3, Hit@5 และ MRR
- เปรียบเทียบคำตอบจาก LLM แบบไม่ใช้ RAG กับแบบใช้ RAG
- แสดงผลการทดลองและผลการประเมินผ่าน Streamlit Dashboard

## ภาพรวมการทำงาน

```text
เอกสาร PDF ใน data/
        |
        v
LlamaParse แปลง PDF เป็นข้อความ
        |
        v
แบ่งข้อความเป็น Chunk
        |
        v
BAAI/bge-m3 สร้าง Embedding
        |
        v
บันทึกลง ChromaDB
        |
        +-------------------------------+
        |                               |
        v                               v
ค้นหาบริบทสำหรับคำถาม          ประเมินผล Retrieval
        |                               |
        v                               v
Gemini สร้างคำตอบ              บันทึกผลใน outputs/
        |
        v
แสดงผลผ่าน Streamlit
```

## เทคโนโลยีที่ใช้

- Python
- ChromaDB สำหรับจัดเก็บและค้นหาเวกเตอร์
- Sentence Transformers และ `BAAI/bge-m3` สำหรับสร้าง Embedding
- LlamaParse สำหรับอ่านข้อความจาก PDF
- Google Gemini รุ่น `gemini-3.1-flash-lite` สำหรับสร้างคำตอบ
- Streamlit สำหรับสร้าง Dashboard
- Pandas สำหรับจัดการข้อมูลผลการประเมิน

## การตั้งค่าหลักของระบบ

| รายการ | ค่าปัจจุบัน |
|---|---|
| พาธฐานข้อมูลเวกเตอร์ | `./chroma_db` |
| ชื่อ ChromaDB Collection | `rag_evaluation_512` |
| โมเดล Embedding | `BAAI/bge-m3` |
| ขนาด Chunk | 512 ตัวอักษร |
| ระยะซ้อนทับระหว่าง Chunk | 50 ตัวอักษร |
| โมเดลสร้างคำตอบ | `gemini-3.1-flash-lite` |
| จำนวนผลลัพธ์สำหรับประเมิน Retrieval | 5 รายการ |
| จำนวนคำถามสำหรับเปรียบเทียบ LLM | 10 ข้อแรก |

## โครงสร้างโครงการ

```text
mini-rag-evaluation/
|-- app.py                         # Streamlit Dashboard
|-- requirements.txt              # รายการไลบรารี Python
|-- .env.example                  # ตัวอย่างตัวแปรสภาพแวดล้อม
|-- data/                         # เอกสาร PDF ต้นฉบับ
|-- evaluation_set/
|   `-- eval_question.json        # ชุดคำถามและคำตอบอ้างอิง
|-- src/
|   |-- ingest.py                 # อ่าน PDF และสร้าง ChromaDB
|   |-- generate_eval_set.py      # สร้างแม่แบบชุดประเมิน
|   |-- evaluate_retrieval.py     # ประเมินคุณภาพการค้นคืน
|   `-- evaluate_llm.py           # เปรียบเทียบ No-RAG กับ RAG
|-- chroma_db/                    # ฐานข้อมูลเวกเตอร์ที่ระบบสร้างขึ้น
`-- outputs/                      # ผลการประเมินที่ระบบสร้างขึ้น
```

ไดเรกทอรี `chroma_db/` และ `outputs/` เป็นข้อมูลที่สร้างใหม่ได้ จึงถูกกำหนดให้ Git ไม่ติดตาม ส่วนเอกสารใน `data/` และชุดคำถามใน `evaluation_set/` เป็นข้อมูลนำเข้าและควรจัดเก็บไว้ในโครงการ

## ข้อกำหนดเบื้องต้น

- Python 3.10 หรือใหม่กว่า
- อินเทอร์เน็ตสำหรับดาวน์โหลดโมเดล `BAAI/bge-m3` และเรียกใช้บริการภายนอก
- Llama Cloud API Key สำหรับ LlamaParse
- Google Gemini API Key สำหรับการสร้างคำตอบและการประเมิน LLM

การเรียกใช้ LlamaParse และ Google Gemini อาจมีข้อจำกัดด้านโควตาหรือค่าใช้จ่ายตามบัญชีและบริการที่ใช้งาน

## การติดตั้ง

### 1. สร้าง Virtual Environment

บน Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

บน macOS หรือ Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. ติดตั้งไลบรารี

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

หากพบข้อผิดพลาดว่าไม่พบโมดูล `llama_parse` หรือ `langchain_text_splitters` ให้ติดตั้งแพ็กเกจที่เกี่ยวข้องเพิ่มเติม:

```bash
pip install llama-parse langchain-text-splitters
```

## การตั้งค่าตัวแปรสภาพแวดล้อม

คัดลอก `.env.example` เป็น `.env`

บน Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

บน macOS หรือ Linux:

```bash
cp .env.example .env
```

กำหนด API Key ในไฟล์ `.env` ดังนี้:

```dotenv
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
GEMINI_API_KEY=your_gemini_api_key
```

ห้าม Commit ไฟล์ `.env` หรือเผยแพร่ API Key ในระบบควบคุมเวอร์ชัน ไฟล์ `.gitignore` ของโครงการได้ตั้งค่าให้ละเว้นไฟล์นี้แล้ว

## วิธีใช้งาน

ควรรันคำสั่งทั้งหมดจากไดเรกทอรีรากของโครงการ เพื่อให้พาธแบบสัมพัทธ์ เช่น `data/`, `chroma_db/` และ `outputs/` ทำงานถูกต้อง

### ขั้นตอนที่ 1: เตรียมเอกสาร

นำไฟล์ PDF ที่ต้องการใช้งานไปไว้ในไดเรกทอรี `data/` ระบบจะอ่านไฟล์นามสกุล `.pdf` ทุกไฟล์ในไดเรกทอรีนี้

### ขั้นตอนที่ 2: สร้างฐานข้อมูลเวกเตอร์

```bash
python src/ingest.py
```

สคริปต์จะดำเนินการดังนี้:

1. อ่าน PDF ด้วย LlamaParse
2. แบ่งข้อความด้วย `RecursiveCharacterTextSplitter`
3. กำหนด `chunk_size` เท่ากับ 512 และ `chunk_overlap` เท่ากับ 50
4. สร้าง Embedding ด้วยโมเดล `BAAI/bge-m3`
5. บันทึกข้อมูลลง Collection ชื่อ `rag_evaluation_512` ใน `chroma_db/`

รหัสของแต่ละ Chunk มีรูปแบบดังนี้:

```text
ชื่อไฟล์.pdf_page_หมายเลขหน้า_chunk_ลำดับ
```

ตัวอย่าง:

```text
Mock_up_Data.pdf_page_1_chunk_1
```

ข้อควรระวัง: สคริปต์ใช้ `collection.add()` หากรันซ้ำกับเอกสารเดิมโดยไม่ลบฐานข้อมูลเดิม อาจเกิดข้อผิดพลาดจากรหัส Chunk ซ้ำ หากต้องการสร้างดัชนีใหม่ทั้งหมด ให้สำรองข้อมูลที่จำเป็นและลบไดเรกทอรี `chroma_db/` ก่อนรันอีกครั้ง

### ขั้นตอนที่ 3: เตรียมชุดประเมิน

หากยังไม่มีไฟล์ `evaluation_set/eval_question.json` ให้สร้างแม่แบบด้วยคำสั่ง:

```bash
python src/generate_eval_set.py
```

สคริปต์จะแสดงตัวอย่าง Chunk จำนวนไม่เกิน 5 รายการ และสร้างคำถามตัวอย่าง 30 ข้อเมื่อยังไม่มีไฟล์ชุดประเมิน จากนั้นต้องแก้ไขคำถาม รหัส Chunk ที่คาดหวัง และคำตอบอ้างอิงให้ตรงกับข้อมูลจริง

รูปแบบข้อมูลของคำถามแต่ละข้อ:

```json
{
  "id": "q1",
  "question": "คำถามที่ต้องการประเมิน",
  "expected_chunk_ids": [
    "Mock_up_Data.pdf_page_1_chunk_1"
  ],
  "reference_answer": "คำตอบอ้างอิงที่ตรวจสอบแล้ว"
}
```

ความหมายของแต่ละฟิลด์:

- `id`: รหัสคำถามที่ไม่ซ้ำกัน
- `question`: คำถามที่ใช้ค้นหาและสร้างคำตอบ
- `expected_chunk_ids`: รายการรหัส Chunk ที่ควรถูกค้นพบ
- `reference_answer`: คำตอบอ้างอิงสำหรับเปรียบเทียบคุณภาพคำตอบ

หากไฟล์ชุดประเมินมีอยู่แล้ว สคริปต์จะไม่เขียนทับไฟล์เดิม

### ขั้นตอนที่ 4: ประเมินการค้นคืน

```bash
python src/evaluate_retrieval.py
```

ผลลัพธ์จะถูกบันทึกที่ `outputs/retrieval_metrics.json` โดยมีทั้งค่าสรุปและรายละเอียดของคำถามแต่ละข้อ

เมตริกที่ใช้มีความหมายดังนี้:

- `Hit@1`: สัดส่วนคำถามที่พบ Chunk ที่คาดหวังในผลลัพธ์อันดับแรก
- `Hit@3`: สัดส่วนคำถามที่พบ Chunk ที่คาดหวังภายใน 3 อันดับแรก
- `Hit@5`: สัดส่วนคำถามที่พบ Chunk ที่คาดหวังภายใน 5 อันดับแรก
- `MRR`: ค่าเฉลี่ยส่วนกลับของอันดับแรกที่พบ Chunk ที่คาดหวัง ค่ายิ่งใกล้ 1 ยิ่งดี

ค่า Hit@K ในไฟล์ผลลัพธ์แสดงเป็นเปอร์เซ็นต์ ส่วน MRR มีช่วงค่าตั้งแต่ 0 ถึง 1

### ขั้นตอนที่ 5: เปรียบเทียบคำตอบ No-RAG กับ RAG

```bash
python src/evaluate_llm.py
```

สคริปต์จะประเมินคำถาม 10 ข้อแรกจากชุดประเมินตามค่าเริ่มต้น โดยสร้างคำตอบสองแบบ:

- `answer_no_rag`: Gemini ตอบจากความรู้ของโมเดลโดยไม่มีบริบทจากเอกสาร
- `answer_rag`: Gemini ตอบโดยใช้ Chunk จำนวน 5 รายการที่ค้นคืนจาก ChromaDB

คำตอบทั้งสองแบบสร้างด้วยโมเดล `gemini-3.1-flash-lite` ผลลัพธ์จะถูกบันทึกที่ `outputs/llm_eval_results.json` พร้อมคำตอบอ้างอิงและบริบทที่ใช้

การประเมินส่วนนี้เป็นการสร้างผลลัพธ์สำหรับเปรียบเทียบ ยังไม่มีการคำนวณคะแนนความถูกต้องหรือความสอดคล้องโดยอัตโนมัติ ผู้ประเมินต้องตรวจคำตอบเทียบกับ `reference_answer` และ `context_used`

### ขั้นตอนที่ 6: เปิด Dashboard

```bash
streamlit run app.py
```

จากนั้นเปิด URL ที่ Streamlit แสดงใน Terminal โดยทั่วไปคือ:

```text
http://localhost:8501
```

Dashboard แบ่งเป็นสองส่วน:

1. Live RAG Demo สำหรับป้อนคำถาม เลือกจำนวน Chunk ระหว่าง 1 ถึง 5 และดูคำตอบพร้อมแหล่งข้อมูล
2. Evaluation Results สำหรับดูค่า Hit@K, MRR และผลเปรียบเทียบคำตอบ

ต้องสร้าง `chroma_db/` ก่อนเปิด Dashboard มิฉะนั้นแอปจะไม่พบ Collection `rag_evaluation_512`

## ลำดับคำสั่งสำหรับการใช้งานครั้งแรก

```bash
pip install -r requirements.txt
python src/ingest.py
python src/evaluate_retrieval.py
python src/evaluate_llm.py
streamlit run app.py
```

ไฟล์ `evaluation_set/eval_question.json` มีอยู่ในโครงการแล้ว จึงไม่จำเป็นต้องรัน `generate_eval_set.py` เว้นแต่ต้องการสร้างชุดประเมินใหม่

## ไฟล์ผลลัพธ์

### `outputs/retrieval_metrics.json`

ประกอบด้วย:

- `summary`: การตั้งค่า จำนวนคำถาม ค่า Hit@K และ MRR
- `details`: คำถาม รหัส Chunk ที่คาดหวัง รหัสที่ค้นคืน อันดับ และ Reciprocal Rank

### `outputs/llm_eval_results.json`

ประกอบด้วยรายการผลประเมินภายใต้ `eval_results` แต่ละรายการมีคำถาม คำตอบอ้างอิง คำตอบแบบ No-RAG คำตอบแบบ RAG และบริบทที่ใช้

## ข้อจำกัดของโค้ดปัจจุบัน

- `app.py` และ `src/evaluate_llm.py` ใช้โมเดล `gemini-3.1-flash-lite` เหมือนกัน จึงต้องตรวจสอบว่าบัญชีและ API Key ที่กำหนดมีสิทธิ์เรียกใช้โมเดลนี้
- `src/evaluate_llm.py` บันทึกข้อมูลเป็นออบเจ็กต์ที่มีคีย์ `eval_results` และใช้ชื่อฟิลด์ `answer_rag`
- ส่วนแสดงผล LLM ใน `app.py` คาดหวังข้อมูลระดับบนสุดเป็นรายการ และคาดหวังฟิลด์ชื่อ `answer_with_rag` ทำให้ผลจากสคริปต์ปัจจุบันอาจยังไม่แสดงใน Dashboard จนกว่าจะปรับรูปแบบข้อมูลให้ตรงกัน
- การประเมิน LLM ยังเป็นการเปรียบเทียบเชิงคุณภาพ ไม่มีเมตริกอัตโนมัติ เช่น Faithfulness, Answer Relevance หรือ Correctness
- การอ่าน PDF ขึ้นอยู่กับคุณภาพของเอกสารและผลจาก LlamaParse เอกสารสแกน ตาราง หรือรูปแบบซับซ้อนอาจให้ข้อความไม่สมบูรณ์
- การสร้าง Embedding ครั้งแรกต้องดาวน์โหลดโมเดลและอาจใช้เวลา รวมถึงต้องใช้หน่วยความจำตามขนาดของเอกสาร

## การแก้ปัญหาเบื้องต้น

### ไม่พบ Collection ใน ChromaDB

ตัวอย่างข้อผิดพลาด:

```text
Collection [rag_evaluation_512] does not exist
```

ให้ตรวจสอบว่าได้รันคำสั่งต่อไปนี้สำเร็จแล้ว:

```bash
python src/ingest.py
```

### ไม่พบ API Key

ตรวจสอบว่าไฟล์ `.env` อยู่ที่ไดเรกทอรีรากของโครงการ และมีตัวแปรต่อไปนี้:

```dotenv
LLAMA_CLOUD_API_KEY=...
GEMINI_API_KEY=...
```

หลังแก้ไข `.env` ให้หยุดและเปิดโปรแกรมใหม่เพื่อโหลดค่าล่าสุด

### ไม่พบไฟล์ PDF

ตรวจสอบว่ามีไฟล์นามสกุล `.pdf` อย่างน้อยหนึ่งไฟล์ใน `data/` และรันคำสั่งจากไดเรกทอรีรากของโครงการ

### รหัส Chunk ซ้ำ

หากรัน `ingest.py` ซ้ำกับข้อมูลเดิม ChromaDB อาจปฏิเสธรหัสที่มีอยู่แล้ว ให้สำรองข้อมูลที่จำเป็น ลบ `chroma_db/` แล้วสร้างฐานข้อมูลใหม่

### Dashboard ไม่แสดงผลการประเมิน

ตรวจสอบว่าได้รันสคริปต์ประเมินและมีไฟล์ต่อไปนี้:

```text
outputs/retrieval_metrics.json
outputs/llm_eval_results.json
```

สำหรับผลประเมิน LLM ให้ตรวจสอบข้อจำกัดเรื่องโครงสร้าง JSON และชื่อฟิลด์ที่อธิบายไว้ในหัวข้อข้อจำกัดของโค้ดปัจจุบัน

## แนวทางการทดลองเพิ่มเติม

สามารถปรับค่าต่อไปนี้เพื่อเปรียบเทียบประสิทธิภาพของระบบ:

- โมเดล Embedding
- ขนาด Chunk และระยะซ้อนทับ
- จำนวนผลลัพธ์ Top K
- รูปแบบ Prompt
- โมเดลที่ใช้สร้างคำตอบ
- จำนวนและความหลากหลายของคำถามในชุดประเมิน

เมื่อเปลี่ยนการตั้งค่าที่มีผลต่อ Chunk หรือ Embedding ควรสร้าง `chroma_db/` ใหม่ และปรับ `expected_chunk_ids` ในชุดประเมินให้ตรงกับรหัส Chunk ชุดใหม่ก่อนคำนวณเมตริกอีกครั้ง
