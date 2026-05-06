import os
import json
import requests
import pandas as pd
from additional_prompt import COURSE, INSTRUCTION
from fetch_student_data import get_student_context
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import create_client
from dotenv import load_dotenv
from irrCAC.raw import CAC

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

if not all([SUPABASE_URL, SUPABASE_KEY, OPENROUTER_API_KEY]):
    raise ValueError("Missing environment variables.")

API_URL = 'https://openrouter.ai/api/v1/chat/completions'
EVAL_MODEL = 'meta-llama/llama-3.3-70b-instruct'

HEADERS = {
    'Authorization': f'Bearer {OPENROUTER_API_KEY}',
    'Content-Type': 'application/json'
}

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_sample_rows(table_name, limit=75):
    """Fetches a sample of rows from the specified table."""
    resp = supabase.table(table_name).select('*').limit(limit).execute()
    return pd.DataFrame(resp.data or [])

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

def run_gwet_validation(df_results):
    """Runs Gwet's AC2 validation on the results."""
    df_clean = df_results.dropna(subset=['human_score', 'llm_score']).copy()
    df_clean['human_score'] = pd.to_numeric(df_clean['human_score'], errors='coerce')
    df_clean['llm_score'] = pd.to_numeric(df_clean['llm_score'], errors='coerce')
    df_clean = df_clean.dropna(subset=['human_score', 'llm_score'])

    print("\n" + "="*30)
    print("--- Gwet Analysis (Human vs LLM) ---")
    
    if len(df_clean) < 2:
        print("Not enough valid samples for Gwet analysis.")
        return

    df_rater = df_clean[['human_score', 'llm_score']]
    cac = CAC(df_rater, weights='quadratic')
    gwet_result = cac.gwet()

    ac2_score = gwet_result['est']['coefficient_value']
    coef_name = gwet_result['est']['coefficient_name']
    p_value = gwet_result['est']['p_value']

    print(f"Coefficient: {coef_name}")
    print(f"Score: {ac2_score:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print("="*30)

    labels = [1, 2, 3, 4, 5]
    cm = pd.crosstab(
        pd.Categorical(df_clean['human_score'], categories=labels),
        pd.Categorical(df_clean['llm_score'], categories=labels),
        rownames=['Human'],
        colnames=['LLM'],
        dropna=False
    )

    print("\n--- Confusion Matrix ---")
    print(cm)

def run_pipeline():
    print("Fetching 75 samples from chatbot_ratings...")
    df_ratings = fetch_sample_rows('chatbot_ratings', limit=75)
    print(f"Rows fetched: {len(df_ratings)}")

    pairs = []
    print("Enriching samples with student context...")
    for _, row in df_ratings.iterrows():
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
    df_pairs['llm_score'] = None
    df_pairs['llm_reason'] = None

    print(f"Starting sample evaluation for {len(df_pairs)} samples...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_llm_score, row): i for i, row in df_pairs.iterrows()}
        for future in as_completed(futures):
            idx = futures[future]
            score, reason = future.result()
            df_pairs.at[idx, 'llm_score'] = score
            df_pairs.at[idx, 'llm_reason'] = reason
            print(f"Progress: {idx+1}/{len(df_pairs)} evaluated")

    df_final = df_pairs.drop(columns=['student_level'])
    df_final.to_csv('evaluation_results_sample.csv', index=False)
    print("Results saved to evaluation_results_sample.csv")

    run_gwet_validation(df_final)

if __name__ == "__main__":
    run_pipeline()
