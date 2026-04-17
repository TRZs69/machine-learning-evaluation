# Machine Learning Evaluation (LLM-as-a-judge)

Repository ini berisi framework evaluasi kuantitatif untuk Chatbot Levely menggunakan metodologi **LLM-as-a-judge**. 
Tujuan utamanya adalah mengukur kualitas respons chatbot berdasarkan akurasi materi (HCI), personalisasi terhadap level siswa (ELO/Grade), dan nilai pedagogis.

## Alur Evaluasi
1.  **Data Fetching**: Mengambil log percakapan dari Supabase (`chat_sessions`, `chat_messages`).
2.  **Enrichment**: Melengkapi data dengan konteks pendidikan (ELO, Grade, Chapter) dari database MySQL (Aiven) atau tabel ringkasan di Supabase.
3.  **Human Sampling**: (Opsional/Prasyarat) Menggunakan sampel yang sudah dinilai manusia (`human_evaluation_sample.xlsx`) untuk validasi metode.
4.  **LLM Judging**: Mengirimkan pasangan (Pesan User + Balasan Chatbot + Konteks Siswa) ke LLM (via OpenRouter) untuk mendapatkan skor 1-5 dan alasan (rationale).
5.  **Validation**: Menghitung korelasi Pearson dan Quadratic Weighted Kappa antara skor LLM dan skor manusia untuk memastikan keandalan (reliability) evaluator LLM.

---

## Persyaratan
- Python 3.9+
- Akses ke **Supabase** (URL & Service Role Key) untuk log percakapan.
- Akses ke **Database MySQL Levelearn** (Aiven) untuk konteks ELO/Grade (jika menggunakan Notebook).
- **OpenRouter API Key** untuk mengakses model LLM.

### Instalasi
```bash
pip install pandas python-dotenv supabase requests matplotlib seaborn scikit-learn openpyxl pymysql sqlalchemy
```

### Konfigurasi (.env)
Salin `.env.example` menjadi `.env` dan lengkapi:
```ini
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="your-key"
OPENROUTER_API_KEY="sk-or-..."
DATABASE_URL="mysql+pymysql://user:pass@host:3306/db"
```

---

## Cara Menjalankan

### Opsi 1: Interactive Notebook (Google Colab / VSCode)
Gunakan `chatbot_ml_evaluation.ipynb` untuk eksplorasi data dan eksperimen prompt.
- **Model Default**: `nvidia/nemotron-3-super-120b-a12b:free` (Free tier).
- **Kelebihan**: Visualisasi interaktif, mudah dijalankan di Colab, mendukung fetch langsung dari MySQL Aiven.
- **Langkah**: Jalankan sel secara berurutan. Notebook mendukung checkpointing di Google Drive.

### Opsi 2: Automated Pipeline (Local Script)
Gunakan `run_full_pipeline.py` untuk menjalankan evaluasi skala besar secara otomatis.
- **Model Default**: `meta-llama/llama-3.3-70b-instruct` (Model berbayar, lebih stabil).
- **Fitur**: Batch processing (concurrent), checkpointing otomatis (`pipeline_checkpoint.csv`), dan kalkulasi metrik validasi di akhir.
- **Cara Menjalankan**:
  ```bash
  python run_full_pipeline.py
  ```
  *Pastikan `human_evaluation_sample.xlsx` sudah tersedia jika ingin melakukan validasi otomatis.*

---

## Skrip Pembantu Lainnya
- `check_results.py`: Ringkasan cepat distribusi skor dari hasil evaluasi terakhir.
- `estimate_cost.py`: Estimasi biaya API OpenRouter berdasarkan jumlah token dan jumlah data.
- `debug_scores.py`: Mencari baris dengan perbedaan skor (gap) besar antara manusia dan LLM untuk debugging prompt.
- `test_new_prompt.py`: Uji coba prompt baru pada sebagian kecil data sebelum menjalankan full pipeline.

## Interpretasi Hasil
Kualitas model evaluator (LLM) dinilai berdasarkan:
- **Pearson Correlation**: Mengukur korelasi linear (Target: > 0.7).
- **Quadratic Weighted Kappa**: Mengukur tingkat kesepakatan (agreement) yang mempertimbangkan besarnya jarak kesalahan (Target: > 0.6 untuk *Substantial Agreement*).

Hasil akhir evaluasi disimpan dalam `full_dataset_evaluation_results.csv` yang berisi kolom `llm_score_full` dan `llm_reason_full`.
