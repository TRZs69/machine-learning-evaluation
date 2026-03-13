# Machine Learning Evaluation (LLM-as-a-judge)

Repository ini berisi script evaluasi kuantitatif untuk Chatbot Levely menggunakan metodologi **LLM-as-a-judge**. 
Tujuan dari evaluasi ini adalah mengukur seberapa relevan balasan chatbot terhadap *progress* belajar siswa dan konteks materi yang sedang dibaca.

## Persyaratan
- Python 3.9+
- Database MySQL Levelearn (Aiven)
- Akses ke Supabase untuk mengambil log percakapan.
- Backend LeveLearn aktif (backend akan memanggil OpenRouter API sebagai proxy).

## Setup
1. Buat virtual environment (opsional) dan install dependencies:
   ```bash
   pip install pandas python-dotenv supabase requests matplotlib seaborn scikit-learn openpyxl pymysql
   ```
2. Salin `.env.example` menjadi `.env` dan isi token untuk Supabase serta URL backend proxy LLM:
   ```
   SUPABASE_URL="your-supabase-url"
   SUPABASE_SERVICE_ROLE_KEY="your-supabase-service-role-key"
   BACKEND_LLM_API_BASE="https://backend-65ah.vercel.app/api/llm/openrouter"
   DATABASE_URL="mysql://username:password@host:3306/database_name"
   ```
3. Pastikan `DATABASE_URL` di `.env` sudah benar dan mengarah ke database MySQL (Aiven) milik Levelearn.
4. Pastikan backend memiliki env OpenRouter (`OPENROUTER_API_KEY`, `OPENROUTER_REFERER`, `OPENROUTER_APP_NAME`) karena notebook tidak memanggil OpenRouter secara langsung.

## Cara Menjalankan Evaluasi
1. Buka `chatbot_ml_evaluation.ipynb` via VSCode atau Jupyter Notebook.
2. Jalankan sel pertama hingga bagian **Evaluasi Menggunakan Multi LLM**.
   > *Proses ini akan mencari model prioritas Kimi 2.5, GLM 5, dan Qwen 3.5 terlebih dulu, lalu fallback ke model free tier lain jika perlu.*
3. Jalankan sel bagian ekspor untuk membuat file `human_evaluation_sample.xlsx`.
4. Buka file Excel tersebut, dan isi nilai manual Anda di kolom `human_score` (1-5).
5. Gabungkan kembali (run bagian terakhir script) untuk melihat tingkat validitas metode ini (Pearson Correlation), agreement antar-model evaluator, dan skor rata-rata Chatbot.
