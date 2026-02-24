# Machine Learning Evaluation (LLM-as-a-judge)

Repository ini berisi script evaluasi kuantitatif untuk Chatbot Levely menggunakan metodologi **LLM-as-a-judge**. 
Tujuan dari evaluasi ini adalah mengukur seberapa relevan balasan chatbot terhadap *progress* belajar siswa dan konteks materi yang sedang dibaca.

## Persyaratan
- Python 3.9+
- Database MySQL Levelearn (Aiven)
- Akses ke Supabase untuk mengambil log percakapan.

## Setup
1. Buat virtual environment (opsional) dan install dependencies:
   ```bash
   pip install pandas python-dotenv supabase google-generativeai matplotlib seaborn scikit-learn openpyxl pymysql
   ```
2. Salin `.env.example` menjadi `.env` dan isi token untuk Supabase dan Gemini API:
   ```
   SUPABASE_URL="your-supabase-url"
   SUPABASE_SERVICE_ROLE_KEY="your-supabase-service-role-key"
   LEVELY_GEMINI_API_KEY="your-gemini-api-key"
   ```
3. Pastikan `DATABASE_URL` di `.env` sudah benar dan mengarah ke database MySQL (Aiven) milik Levelearn.

## Cara Menjalankan Evaluasi
1. Buka `chatbot_ml_evaluation.ipynb` via VSCode atau Jupyter Notebook.
2. Jalankan sel pertama hingga bagian **Evaluasi Menggunakan LLM**.
   > *Proses ini akan mengirim data ke Gemini API untuk dinilai.*
3. Jalankan sel bagian ekspor untuk membuat file `human_evaluation_sample.xlsx`.
4. Buka file Excel tersebut, dan isi nilai manual Anda di kolom `human_score` (1-5).
5. Gabungkan kembali (run bagian terakhir script) untuk melihat tingkat validitas metode ini (Pearson Correlation) dan skor rata-rata Chatbot.
