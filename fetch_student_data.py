import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import urllib.parse as urlparse

load_dotenv()

def get_engine():
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not found in environment variables.")

    url_parts = urlparse.urlparse(DATABASE_URL)
    
    query_params = urlparse.parse_qs(url_parts.query)
    
    problematic_params = ['connection_limit', 'ssl-mode', 'pool_timeoute']
    
    filtered_params = {k: v for k, v in query_params.items() if k not in problematic_params}
    
    base_url = f"{url_parts.scheme}://{url_parts.username}:{url_parts.password}@{url_parts.hostname}:{url_parts.port}{url_parts.path}"
    
    connect_args = {}
    if 'ssl-mode' in query_params or 'ssl_mode' in query_params or 'ssl' in url_parts.query:
        connect_args["ssl"] = {"ca": None} 

    return create_engine(base_url, connect_args=connect_args)

engine = get_engine()

def get_student_context(user_id):
    """
    Fetches student state from MySQL to provide context for Levely.
    """
    context = {}
    
    try:
        with engine.connect() as conn:
            user_query = text("SELECT name, points, elo FROM users WHERE id = :uid")
            user_res = conn.execute(user_query, {"uid": user_id}).fetchone()
            
            if not user_res:
                return None
            
            context['name'] = user_res.name
            context['total_points'] = user_res.points
            context['global_elo'] = user_res.elo

            chapter_query = text("""
                SELECT currentDifficulty, correctStreak, wrongStreak 
                FROM user_chapters 
                WHERE userId = :uid 
                ORDER BY updatedAt DESC LIMIT 1
            """)
            chap_res = conn.execute(chapter_query, {"uid": user_id}).fetchone()
            
            if chap_res:
                context['current_difficulty'] = chap_res.currentDifficulty
                context['streaks'] = {
                    "correct": chap_res.correctStreak,
                    "wrong": chap_res.wrongStreak
                }
            else:
                context['current_difficulty'] = "BEGINNER"
                context['streaks'] = {"correct": 0, "wrong": 0}

            session_query = text("""
                SELECT durationSec FROM user_sessions 
                WHERE userId = :uid 
                ORDER BY loginAt DESC LIMIT 1
            """)
            sess_res = conn.execute(session_query, {"uid": user_id}).fetchone()
            context['last_session_seconds'] = sess_res.durationSec if sess_res else 0

        return context

    except Exception as e:
        print(f"Error fetching from MySQL for user {user_id}: {e}")
        return None
