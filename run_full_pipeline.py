import os
import time
import json
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================================
# 1. CONFIGURATION & SETUP
# ==========================================================

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

if not all([SUPABASE_URL, SUPABASE_KEY, OPENROUTER_API_KEY]):
    raise ValueError("Missing environment variables. Please check .env file.")

# OpenRouter Config
API_URL = 'https://openrouter.ai/api/v1/chat/completions'
EVAL_MODEL = 'meta-llama/llama-3.3-70b-instruct'
SLEEP_SECONDS = 0  # Paid model
BATCH_SIZE = 75    # Concurrent requests
CHECKPOINT_FILE = 'pipeline_checkpoint.csv'

HEADERS = {
    'Authorization': f'Bearer {OPENROUTER_API_KEY}',
    'Content-Type': 'application/json',
    'HTTP-Referer': 'http://localhost',
    'X-Title': 'LeveLearn Evaluation Pipeline'
}

# Supabase Client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================================
# 2. DATA FETCHING & PREPARATION
# ==========================================================

def fetch_all_rows(table_name):
    print(f"🔍 Fetching {table_name}...")
    rows = []
    start = 0
    while True:
        resp = supabase.table(table_name).select('*').range(start, start + 999).execute()
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < 1000:
            break
        start += 1000
    print(f"   → {len(rows)} rows fetched.")
    return pd.DataFrame(rows)

print("=" * 60)
print("STEP 1: FETCHING DATA")
print("=" * 60)

df_sessions = fetch_all_rows('chat_sessions')
df_messages = fetch_all_rows('chat_messages')
df_summaries = fetch_all_rows('student_summaries')

# Build Pairs
print("\n🧩 Building interaction pairs...")
df_merged = pd.merge(
    df_messages, 
    df_sessions[['id', 'user_id']], 
    left_on='session_id', 
    right_on='id', 
    how='left', 
    suffixes=('_msg', '_session')
)
df_merged = df_merged.sort_values(['user_id', 'session_id', 'created_at']).reset_index(drop=True)

pairs = []
current_user_msg = None
for _, row in df_merged.iterrows():
    if row['role'] == 'user':
        current_user_msg = row
    elif row['role'] == 'assistant' and current_user_msg is not None:
        pairs.append({
            'eval_id': f"{row['session_id']}_{row['id_msg']}",
            'user_id': str(row['user_id']),
            'user_message': current_user_msg['content'],
            'chatbot_reply': row['content']
        })
        current_user_msg = None

df_pairs = pd.DataFrame(pairs)
print(f"✅ Extracted {len(df_pairs)} pairs.")

# Enrich with Student Context
print("\n🎓 Enriching with student context...")
df_pairs['user_id'] = df_pairs['user_id'].astype(str)
df_summaries['user_id'] = df_summaries['user_id'].astype(str)

context_cols = ['student_name', 'avg_grade', 'total_points_earned', 'chapters_completed']
available_cols = [c for c in context_cols if c in df_summaries.columns]

if available_cols:
    df_pairs = pd.merge(df_pairs, df_summaries[['user_id'] + available_cols], on='user_id', how='left')

# Create Context Summary
def make_context(row):
    parts = []
    if 'student_name' in row and pd.notna(row['student_name']): parts.append(f"Student: {row['student_name']}")
    if 'avg_grade' in row and pd.notna(row['avg_grade']): parts.append(f"Grade: {int(row['avg_grade'])}%")
    if 'total_points_earned' in row and pd.notna(row['total_points_earned']): parts.append(f"Points: {int(row['total_points_earned'])}")
    return '; '.join(parts) if parts else 'N/A'

df_pairs['educational_context'] = df_pairs.apply(make_context, axis=1)

# ==========================================================
# 3. SAMPLING & HUMAN SCORING
# ==========================================================

print("\n" + "=" * 60)
print("STEP 2: SAMPLING & HUMAN SCORING")
print("=" * 60)

human_sample_file = 'human_evaluation_sample.xlsx'
if not os.path.exists(human_sample_file):
    raise FileNotFoundError(f"{human_sample_file} not found! Run generate_human_sample.py and fill_human_scores.py first.")

df_human = pd.read_excel(human_sample_file)
eval_ids_sample = df_human['eval_id'].tolist()

print(f"📄 Loaded {len(df_human)} human scores.")

# Merge Human Scores into Pairs
df_pairs = pd.merge(df_pairs, df_human[['eval_id', 'score']], on='eval_id', how='left')
df_pairs.rename(columns={'score': 'human_score'}, inplace=True)

# Filter to Sample for Validation
df_sample = df_pairs[df_pairs['human_score'].notna()].copy()
print(f"🔍 Sample size for validation: {len(df_sample)}")

# ==========================================================
# 4. LLM JUDGING (BATCH MODE)
# ==========================================================

# Syllabus for Reference
COURSE_SYLLABUS = """
MATERI REFERENSI (LeveLearn HCI Course Syllabus):
1. Pengantar HCI: Interaksi manusia & komputer. Fokus: usability, usefulness, satisfaction.
2. Usability Heuristics (Nielsen): 10 prinsip (Visibility, Match real world, User control, Consistency, Error prevention, Recognition, Flexibility, Aesthetic, Help users, Documentation).
3. Cognitive Load Theory: Intrinsic (kompleksitas materi), Extraneous (desain buruk), Germane (proses belajar). Desain harus minimize extraneous load.
4. User-Centered Design (UCD): Iteratif melibatkan user (Research -> Design -> Test -> Iterate).
5. Fitts's Law: Waktu gerak pointer tergantung jarak & ukuran target. Target besar & dekat = lebih cepat.
6. Norman's Action Cycle: 7 tahap (Goal -> Intention -> Action -> Execute -> Perceive -> Interpret -> Evaluate). Gap Execution & Evaluation.
7. Mental Model: Representasi user tentang cara kerja sistem. Desain harus match dengan mental model user.
8. Accessibility (WCAG): Perceivable, Operable, Understandable, Robust.
9. User Experience (UX): Meliputi emosi, estetika, dan nilai praktis dari penggunaan sistem.
"""

def build_judge_prompt(row):
    context = row['educational_context']
    elo_band = "N/A"
    grade = "N/A"
    if "Grade:" in context:
        grade = context.split("Grade:")[1].split(";")[0].strip()
    try:
        g = int(grade.replace('%', ''))
        if g >= 90: elo_band = "Mastery"
        elif g >= 80: elo_band = "Advanced"
        elif g >= 65: elo_band = "Intermediate"
        elif g >= 50: elo_band = "Developing Learner"
        else: elo_band = "Beginner"
    except: pass

    return f"""
You are an Educational Evaluator for LeveLearn. Rate the quality of a chatbot response to a student on a 1-5 scale.

REFERENCE MATERIAL:
{COURSE_SYLLABUS}

STUDENT CONTEXT:
- Level: {elo_band} (Grade={grade})
- Summary: {context}

INTERACTION:
User: {row['user_message']}
Chatbot: {row['chatbot_reply']}

Rate on 1-5 scale:
5 = Excellent: accurate to reference material, gives concrete examples, personalized to student level, well-structured
4 = Good: accurate and helpful, minor details may be missing, good personalization
3 = Average: factually correct but generic, lacks depth or personalization
2 = Below average: has factual errors, not personalized, too brief or unfocused
1 = Poor: incorrect, misleading, or harmful

Be fair and discriminating. Not all responses deserve the same score.

Return ONLY valid JSON:
{{"score": <integer 1-5>, "reason": "<reason in Indonesian>"}}
""".strip()

def get_llm_score(row, retries=3):
    prompt = build_judge_prompt(row)
    payload = {
        'model': EVAL_MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.1
    }

    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
            
            # Check HTTP status
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  ⚠️ 429 Rate Limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            elif resp.status_code != 200:
                return None, None, f"API Error {resp.status_code}: {resp.text[:150]}"
            
            # Validate response body is not empty
            response_text = resp.text.strip()
            if not response_text:
                error_msg = "Empty API response (no content returned)"
                if attempt < retries - 1:
                    print(f"  ⚠️ {error_msg}. Retrying ({attempt + 1}/{retries})...")
                    time.sleep(SLEEP_SECONDS * (attempt + 1))
                    continue
                return None, None, error_msg
            
            # Parse JSON safely
            try:
                resp_json = resp.json()
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON in API response: {str(e)[:100]}. Response preview: {response_text[:150]}"
                if attempt < retries - 1:
                    print(f"  ⚠️ {error_msg}. Retrying ({attempt + 1}/{retries})...")
                    time.sleep(SLEEP_SECONDS * (attempt + 1))
                    continue
                return None, None, error_msg
            
            # Validate response structure
            if 'choices' not in resp_json or not resp_json['choices']:
                error_msg = "API response missing 'choices' field"
                if attempt < retries - 1:
                    print(f"  ⚠️ {error_msg}. Retrying ({attempt + 1}/{retries})...")
                    time.sleep(SLEEP_SECONDS * (attempt + 1))
                    continue
                return None, None, error_msg
            
            # Extract content safely
            try:
                content = resp_json['choices'][0]['message']['content']
            except (KeyError, IndexError, TypeError) as e:
                error_msg = f"Failed to extract content: {str(e)}. Response: {str(resp_json)[:200]}"
                if attempt < retries - 1:
                    print(f"  ⚠️ {error_msg}. Retrying ({attempt + 1}/{retries})...")
                    time.sleep(SLEEP_SECONDS * (attempt + 1))
                    continue
                return None, None, error_msg
            
            # Validate content is not empty
            if not content or not content.strip():
                error_msg = "LLM returned empty content"
                if attempt < retries - 1:
                    print(f"  ⚠️ {error_msg}. Retrying ({attempt + 1}/{retries})...")
                    time.sleep(SLEEP_SECONDS * (attempt + 1))
                    continue
                return None, None, error_msg
            
            # Clean JSON from markdown
            content_cleaned = content.strip()
            if '```json' in content_cleaned:
                try:
                    content_cleaned = content_cleaned.split('```json')[1].split('```')[0]
                except IndexError:
                    pass  # Fall through to normal parsing
            elif '```' in content_cleaned:
                try:
                    content_cleaned = content_cleaned.split('```')[1]
                except IndexError:
                    pass  # Fall through to normal parsing
            
            content_cleaned = content_cleaned.strip()
            
            # Validate cleaned content is not empty
            if not content_cleaned:
                error_msg = "No JSON content found in LLM response"
                if attempt < retries - 1:
                    print(f"  ⚠️ {error_msg}. Retrying ({attempt + 1}/{retries})...")
                    time.sleep(SLEEP_SECONDS * (attempt + 1))
                    continue
                return None, None, error_msg
            
            # Parse JSON safely
            try:
                data = json.loads(content_cleaned)
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse JSON: {str(e)[:100]}. Content: {content_cleaned[:200]}"
                if attempt < retries - 1:
                    print(f"  ⚠️ {error_msg}. Retrying ({attempt + 1}/{retries})...")
                    time.sleep(SLEEP_SECONDS * (attempt + 1))
                    continue
                return None, None, error_msg
            
            # Validate required fields
            if 'score' not in data:
                error_msg = f"JSON missing 'score' field: {str(data)[:200]}"
                if attempt < retries - 1:
                    print(f"  ⚠️ {error_msg}. Retrying ({attempt + 1}/{retries})...")
                    time.sleep(SLEEP_SECONDS * (attempt + 1))
                    continue
                return None, None, error_msg

            # Validate score is valid integer
            try:
                score = int(data['score'])
            except (ValueError, TypeError) as e:
                error_msg = f"Invalid score value: {data['score']} ({str(e)})"
                if attempt < retries - 1:
                    print(f"  ⚠️ {error_msg}. Retrying ({attempt + 1}/{retries})...")
                    time.sleep(SLEEP_SECONDS * (attempt + 1))
                    continue
                return None, None, error_msg
            
            return score, data.get('reason', ''), None
            
        except requests.exceptions.Timeout:
            error_msg = "Request timeout (60s)"
            if attempt < retries - 1:
                print(f"  ⚠️ {error_msg}. Retrying ({attempt + 1}/{retries})...")
                time.sleep(SLEEP_SECONDS * (attempt + 1))
                continue
            return None, None, error_msg
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)[:100]}"
            if attempt < retries - 1:
                print(f"  ⚠️ {error_msg}. Retrying ({attempt + 1}/{retries})...")
                time.sleep(SLEEP_SECONDS * (attempt + 1))
                continue
            return None, None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)[:150]}"
            if attempt < retries - 1:
                print(f"  ⚠️ {error_msg}. Retrying ({attempt + 1}/{retries})...")
                time.sleep(SLEEP_SECONDS * (attempt + 1))
                continue
            return None, None, error_msg
    
    return None, None, "Max retries exceeded"

print("\n" + "=" * 60)
print(f"STEP 3: LLM JUDGING (Batch Mode - {EVAL_MODEL}, batch_size={BATCH_SIZE})")
print("=" * 60)

# Merge Checkpoint
df_checkpoint = None
if os.path.exists(CHECKPOINT_FILE):
    print(f"📂 Loading checkpoint: {CHECKPOINT_FILE}")
    df_checkpoint = pd.read_csv(CHECKPOINT_FILE)
    print(f"   → Found {len(df_checkpoint)} existing results.")
else:
    df_checkpoint = pd.DataFrame(columns=['eval_id', 'llm_score_full', 'llm_reason_full'])

# Prepare full dataframe
df_full = df_pairs.copy()

# Merge existing results
if not df_checkpoint.empty:
    df_full = pd.merge(df_full, df_checkpoint[['eval_id', 'llm_score_full', 'llm_reason_full']], on='eval_id', how='left')
else:
    df_full['llm_score_full'] = None
    df_full['llm_reason_full'] = None

# Identify rows to process
mask_done = df_full['llm_score_full'].notna()
rows_to_process = df_full[~mask_done].copy()
already_done = mask_done.sum()

print(f"\n📊 Status:")
print(f"   Total pairs: {len(df_full)}")
print(f"   Already done: {already_done}")
print(f"   To process: {len(rows_to_process)}")

# Batch concurrent processing
start_time = time.time()
total = len(rows_to_process)

for batch_start in range(0, len(rows_to_process), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(rows_to_process))
    batch_rows = rows_to_process.iloc[batch_start:batch_end]
    batch_num = batch_start // BATCH_SIZE + 1
    total_batches = (len(rows_to_process) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n📦 Batch {batch_num}/{total_batches} ({len(batch_rows)} requests)")

    def process_one(idx, row):
        score, reason, err = get_llm_score(row)
        return idx, score, reason, err

    results = []
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = {executor.submit(process_one, idx, row): idx for idx, row in batch_rows.iterrows()}
        for future in as_completed(futures):
            idx, score, reason, err = future.result()
            results.append((idx, score, reason, err))

    # Apply results
    success_count = 0
    for idx, score, reason, err in results:
        if score is not None:
            df_full.loc[idx, 'llm_score_full'] = score
            df_full.loc[idx, 'llm_reason_full'] = reason
            success_count += 1
        else:
            df_full.loc[idx, 'llm_score_full'] = -1
            df_full.loc[idx, 'llm_reason_full'] = f"Failed: {err}"

    # Save checkpoint after each batch
    current_done = df_full[df_full['llm_score_full'].notna() & (df_full['llm_score_full'] != -1)]
    current_done[['eval_id', 'llm_score_full', 'llm_reason_full']].to_csv(CHECKPOINT_FILE, index=False)

    processed_so_far = already_done + batch_end
    elapsed = time.time() - start_time
    rate = processed_so_far / elapsed if elapsed > 0 else 0
    eta = (total - processed_so_far) / rate if rate > 0 else 0
    print(f"  ✅ {success_count}/{len(batch_rows)} succeeded | Done: {processed_so_far}/{total} | Rate: {rate:.1f}/s | ETA: {eta:.0f}s")

print("\n✅ Full Judging Complete.")

# Save Final Results
output_file = 'full_dataset_evaluation_results.csv'
df_full.to_csv(output_file, index=False)
print(f"💾 Results saved to: {output_file}")

# ==========================================================
# 5. COMPARISON (KAPPA & PEARSON) - VALIDATION
# ==========================================================

print("\n" + "=" * 60)
print("STEP 4: VALIDATION RESULTS")
print("=" * 60)

# Reload to ensure we have latest data
df_res = pd.read_csv(output_file)
df_sample_val = df_res[df_res['human_score'].notna()]

df_compare = df_sample_val.dropna(subset=['llm_score_full', 'human_score']).copy()

if len(df_compare) > 1:
    try:
        from sklearn.metrics import cohen_kappa_score
        pearson = df_compare['llm_score_full'].corr(df_compare['human_score'])
        kappa_quad = cohen_kappa_score(df_compare['llm_score_full'], df_compare['human_score'], weights='quadratic')
        kappa_lin = cohen_kappa_score(df_compare['llm_score_full'], df_compare['human_score'], weights='linear')

        print(f"\n📊 COMPARISON RESULTS (Sample of {len(df_compare)}):")
        print(f"   Pearson Correlation:  {pearson:.4f}")
        print(f"   Quadratic Kappa:      {kappa_quad:.4f}")

        # Use linear kappa as primary metric
        if kappa_lin > 0.6:
            print("   ✅ STATUS: SUBSTANTIAL AGREEMENT.")
        elif kappa_lin > 0.4:
            print("   ⚠️  STATUS: MODERATE AGREEMENT.")
        elif kappa_lin > 0.2:
            print("   ⚠️  STATUS: FAIR AGREEMENT")
        else:
            print("   ❌ STATUS: POOR AGREEMENT.")
            
    except ImportError:
        print("⚠️  Scikit-learn not installed. Cannot calculate Kappa.")
        pearson = df_compare['llm_score_full'].corr(df_compare['human_score'])
        print(f"   Pearson Correlation: {pearson:.4f}")
else:
    print("❌ Not enough data to compare.")

print("\n🚀 PIPELINE FINISHED SUCCESSFULLY.")
