import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and "ssl-mode" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("ssl-mode", "ssl_mode")

engine = create_engine(DATABASE_URL)

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
        print(f"Error fetching from MySQL: {e}")
        return None