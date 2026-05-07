import os
import json
import requests
import pandas as pd
from additional_prompt import COURSE, INSTRUCTION
from fetch_student_data import get_student_context
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import create_client
from dotenv import load_dotenv

load_dotenv(override=True)

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

def fetch_messages_and_sessions():
    """Fetches chat messages and sessions from Supabase."""
    print("Fetching chat_messages (<= April 9)...")
    messages = []
    start = 0
    while True:
        # Fetch messages up to April 9, 2026
        resp = supabase.table('chat_messages').select('*').lte('created_at', '2026-04-09T23:59:59').order('created_at').range(start, start + 999).execute()
        batch = resp.data or []
        messages.extend(batch)
        if len(batch) < 1000: break
        start += 1000
    
    print(f"Fetched {len(messages)} messages.")
    
    print("Fetching chat_sessions...")
    sessions = []
    start = 0
    while True:
        resp = supabase.table('chat_sessions').select('id, user_id').range(start, start + 999).execute()
        batch = resp.data or []
        sessions.extend(batch)
        if len(batch) < 1000: break
        start += 1000
    
    print(f"Fetched {len(sessions)} sessions.")
    return pd.DataFrame(messages), pd.DataFrame(sessions)

def pair_messages(df_messages, df_sessions):
    """Pairs user messages with the subsequent assistant reply."""
    session_to_user = dict(zip(df_sessions['id'], df_sessions['user_id']))
    pairs = []
    
    # Sort messages by session and creation time to ensure correct pairing
    df_messages = df_messages.sort_values(['session_id', 'created_at'])
    
    for session_id, group in df_messages.groupby('session_id'):
        user_id = session_to_user.get(session_id)
        msgs = group.to_dict('records')
        for i in range(len(msgs) - 1):
            # A pair is a user message followed by an assistant message
            if msgs[i]['role'] == 'user' and msgs[i+1]['role'] == 'assistant':
                pairs.append({
                    'eval_id': msgs[i+1]['id'],
                    'user_id': user_id,
                    'user_message': msgs[i]['content'],
                    'chatbot_reply': msgs[i+1]['content'],
                    'human_score': None # Placeholder if no rating exists
                })
    return pd.DataFrame(pairs)

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

def fetch_ratings():
    """Fetches existing ratings for exclusion."""
    print("Fetching chatbot_ratings for exclusion...")
    ratings = []
    start = 0
    while True:
        resp = supabase.table('chatbot_ratings').select('user_id, user_request, bot_response').range(start, start + 999).execute()
        batch = resp.data or []
        ratings.extend(batch)
        if len(batch) < 1000: break
        start += 1000
    print(f"Fetched {len(ratings)} existing ratings.")
    return pd.DataFrame(ratings)

def run_pipeline():
    df_messages, df_sessions = fetch_messages_and_sessions()
    
    print("Pairing user-assistant messages...")
    df_pairs_raw = pair_messages(df_messages, df_sessions)
    print(f"Total pairs identified from logs: {len(df_pairs_raw)}")

    # Fetch and filter out existing ratings
    df_ratings = fetch_ratings()
    
    def normalize(text):
        if not isinstance(text, str): return ""
        return " ".join(text.strip().split())

    rating_keys = set()
    for _, r in df_ratings.iterrows():
        key = (r['user_id'], normalize(r['user_request']), normalize(r['bot_response']))
        rating_keys.add(key)

    initial_count = len(df_pairs_raw)
    filtered_pairs = []
    for _, p in df_pairs_raw.iterrows():
        key = (p['user_id'], normalize(p['user_message']), normalize(p['chatbot_reply']))
        if key not in rating_keys:
            filtered_pairs.append(p)
        
    df_pairs_raw = pd.DataFrame(filtered_pairs)
    print(f"Filtered pairs: {len(df_pairs_raw)} (Excluded {initial_count - len(df_pairs_raw)} pairs already present in chatbot_ratings)")

    pairs = []
    print("Enriching samples with student context (this may take a while)...")
    
    # Cache student context to avoid redundant slow DB calls
    student_cache = {}
    total_pairs = len(df_pairs_raw)
    
    for i, row in df_pairs_raw.iterrows():
        user_id = row['user_id']
        if user_id not in student_cache:
            student_cache[user_id] = get_student_context(user_id)
        
        student_ctx = student_cache[user_id]
        
        pairs.append({
            'eval_id': row['eval_id'],
            'user_id': user_id,
            'user_message': row['user_message'],
            'chatbot_reply': row['chatbot_reply'],
            'human_score': row['human_score'],
            'student_level': student_ctx.get('current_difficulty') if student_ctx else 'N/A',
            'student_elo': student_ctx.get('global_elo') if student_ctx else 'N/A'
        })
        
        if (i + 1) % 10 == 0 or (i + 1) == total_pairs:
            print(f"Enrichment progress: {i+1}/{total_pairs} pairs processed")

    df_pairs = pd.DataFrame(pairs)
    df_pairs['llm_score'] = None
    df_pairs['llm_reason'] = None

    print(f"Starting FULL evaluation for {len(df_pairs)} rows...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_llm_score, row): i for i, row in df_pairs.iterrows()}
        for future in as_completed(futures):
            idx = futures[future]
            score, reason = future.result()
            df_pairs.at[idx, 'llm_score'] = score
            df_pairs.at[idx, 'llm_reason'] = reason
            if (idx + 1) % 10 == 0 or (idx + 1) == len(df_pairs):
                print(f"Progress: {idx+1}/{len(df_pairs)} evaluated")

    df_final = df_pairs.drop(columns=['student_level'])
    df_final.to_csv('evaluation_results_full.csv', index=False)
    print("Done. Results saved to evaluation_results_full.csv")

if __name__ == "__main__":
    run_pipeline()
