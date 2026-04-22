import os
import json
import requests
import pandas as pd
from additional_prompt import COURSE, INSTRUCTION
from fetch_student_data import get_student_context
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

if not all([SUPABASE_URL, SUPABASE_KEY, OPENROUTER_API_KEY]):
    raise ValueError("Missing environment variables.")

API_URL = 'https://openrouter.ai/api/v1/chat/completions'
EVAL_MODEL = 'meta-llama/llama-3.3-70b-instruct'
BATCH_SIZE = 50 

HEADERS = {
    'Authorization': f'Bearer {OPENROUTER_API_KEY}',
    'Content-Type': 'application/json'
}

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_all_rows(table_name):
    rows = []
    start = 0
    while True:
        resp = supabase.table(table_name).select('*').range(start, start + 999).execute()
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < 1000: break
        start += 1000
    return pd.DataFrame(rows)

print("Fetching 75 samples from chatbot_ratings...")
df_ratings = fetch_all_rows('chatbot_ratings')

pairs = []
print("Enriching samples with student context...")
for _, row in df_ratings.iterrows():
    # Fetch student context from MySQL
    student_ctx = get_student_context(row['user_id'])
    
    pairs.append({
        'eval_id': row['id'],
        'user_id': row['user_id'],
        'user_message': row['user_request'],
        'chatbot_reply': row['bot_response'],
        'human_score': row['rating'],
        'student_level': student_ctx.get('current_difficulty') if student_ctx else 'N/A',
        'student_elo': student_ctx.get('global_elo') if student_ctx else 'N/A'
    })

df_pairs = pd.DataFrame(pairs)

def build_judge_prompt(row):
    return f"""
    {INSTRUCTION}

    STUDENT CONTEXT:
    - Difficulty Level: {row['student_level']}
    - ELO Score: {row['student_elo']}

    REFERENCE MATERIAL:
    {COURSE}

    INTERACTION:
    User: {row['user_message']}
    Chatbot: {row['chatbot_reply']}

    Return ONLY valid JSON:
    {{"score": <int>, "reason": "<str>"}}
    """.strip()

def get_llm_score(row):
    prompt = build_judge_prompt(row)
    
    try:
        resp = requests.post(API_URL, headers=HEADERS, timeout=60, json={
            'model': EVAL_MODEL,
            'messages': [{'role': 'user', 'content': prompt}],
            'response_format': {'type': 'json_object'}
        })
        
        if resp.status_code != 200:
            return None, f"Error {resp.status_code}: {resp.text}"

        content = resp.json()['choices'][0]['message']['content']
        res = json.loads(content)
        return res.get('score'), res.get('reason')
    except Exception as e:
        return None, str(e)

df_pairs['llm_score'] = None
df_pairs['llm_reason'] = None

print(f"Starting evaluation for {len(df_pairs)} samples...")

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(get_llm_score, row): i for i, row in df_pairs.iterrows()}
    for future in as_completed(futures):
        idx = futures[future]
        score, reason = future.result()
        df_pairs.at[idx, 'llm_score'] = score
        df_pairs.at[idx, 'llm_reason'] = reason
        print(f"Progress: {idx+1}/{len(df_pairs)} evaluated")

df_final = df_pairs.drop(columns=['student_level'])
df_final.to_csv('evaluation_results.csv', index=False)
print("Done. Results saved to evaluation_results.csv")