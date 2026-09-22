# 🌿 Masvia RAG Chatbot (WhatsApp Customer Service)

Sistem Customer Service AI berbasis **Retrieval-Augmented Generation (RAG)** untuk produk minuman herbal antidiabetes **Jamu Celup Masvia**. Ditenagai oleh **Groq Cloud API** dengan model pilihan `qwen/qwen3.8-27b`, embedding lokal `all-MiniLM-L6-v2`, dan database vektor **ChromaDB**.

---

## 🛠️ Tech Stack (100% Free & Open-Source)

* **LLM Engine:** [Groq Cloud API](https://console.groq.com) (`qwen/qwen3.8-27b` untuk inferensi bahasa Indonesia yang natural, cepat, dan santun).
* **Embeddings:** `all-MiniLM-L6-v2` via HuggingFace (berjalan 100% lokal di CPU — tanpa kuota API, tanpa biaya).
* **Vector Database:** [ChromaDB](https://www.trychroma.com/) (tersimpan lokal di `./chroma_db`).
* **Backend API & Webhook:** [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn.
* **Orchestration:** LangChain Core, LangChain Community, LangChain Groq.
* **Knowledge Base:** Single Source of Truth [masvia-rag-knowledge-base.md](masvia-rag-knowledge-base.md).

---

## 📁 Struktur File Proyek

```text
D:\chatbot-masvia\
├── masvia-rag-knowledge-base.md   # Dokumen pengetahuan lengkap (Fakta, Kandungan, Q&A Q1-Q27)
├── ingest.py                      # Script chunking semantik cerdas & vectorization
├── rag_engine.py                  # Core engine RAG, persona 'Masvia Care', guardrails & Groq LLM
├── query.py                       # CLI interaktif untuk simulasi & tes chat di terminal
├── app.py                         # Server FastAPI & Webhook WhatsApp Cloud API
├── requirements.txt               # Daftar dependensi library Python
├── .env                           # File konfigurasi API Key, Model, dan parameter
├── .env.example                   # Template konfigurasi
└── chroma_db/                     # Direktori penyimpanan vektor Chroma (dibuat oleh ingest.py)
```

---

## 🚀 Panduan Penggunaan (Quickstart)

### 1. Ingestion Dokumen ke Database Vektor
Jalankan perintah ini untuk memecah `masvia-rag-knowledge-base.md` menjadi chunk-chunk cerdas dan menyimpannya ke ChromaDB:
```bash
.\venv\Scripts\python.exe ingest.py
```

### 2. Uji Coba Chat Interaktif di Terminal (CLI)
Anda dapat langsung berinteraksi dengan chatbot Masvia Care melalui terminal:
```bash
.\venv\Scripts\python.exe query.py
```
*Contoh pertanyaan:*
- *"Apa itu Jamu Masvia?"*
- *"Berapa harganya dan isi berapa sachet?"*
- *"Apakah jamu ini pahit?"*
- *"Ada promo gratis ongkir ke kampus UNNES gak?"*
- *"Saya mau pesan 2 pouch, bagaimana caranya?"*

### 3. Menjalankan Server API & Webhook
Jalankan server FastAPI:
```bash
.\venv\Scripts\python.exe app.py
```
Server akan aktif di `http://localhost:8000`. Dokumentasi interaktif Swagger UI dapat diakses di `http://localhost:8000/docs`.

---

## 💬 Menghubungkan ke WhatsApp (Meta Cloud API)

1. **Jalankan tunnel lokal** (misal menggunakan Cloudflare Tunnel atau ngrok):
   ```bash
   ngrok http 8000
   ```
2. Salin URL publik (contoh: `https://abcd-1234.ngrok-free.app/webhook`).
3. Buka **Meta for Developers** > WhatsApp > Configuration:
   - **Callback URL:** `https://abcd-1234.ngrok-free.app/webhook`
   - **Verify Token:** Samakan dengan `WHATSAPP_VERIFY_TOKEN` di `.env` (default: `masvia_secret_verify_token`).
4. Isi `WHATSAPP_TOKEN` dan `WHATSAPP_PHONE_NUMBER_ID` di `.env` untuk mengaktifkan pengiriman pesan balasan otomatis.
