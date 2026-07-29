import sqlite3
import pandas as pd
from datetime import datetime

DB_FILE = "workouts.db"

def init_db():
    """Initializes the SQLite database and creates the history table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workout_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            time TEXT,
            exercise TEXT,
            reps INTEGER,
            duration REAL,
            avg_fps REAL
        )
    """)
    conn.commit()
    conn.close()

def log_workout(exercise: str, reps: int, duration: float, avg_fps: float) -> bool:
    """
    Saves a workout record into the database.
    Only logs if reps > 0 or duration is significant.
    """
    if reps <= 0 and duration < 2.0:
        return False
        
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO workout_history (date, time, exercise, reps, duration, avg_fps)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (date_str, time_str, exercise, reps, round(duration, 1), round(avg_fps, 1)))
    conn.commit()
    conn.close()
    return True

def get_workout_history() -> pd.DataFrame:
    """Fetches all workout history records as a Pandas DataFrame."""
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query("""
            SELECT date as Date, 
                   time as Time, 
                   exercise as Exercise, 
                   reps as Reps, 
                   duration as 'Duration (s)', 
                   avg_fps as 'Avg FPS' 
            FROM workout_history 
            ORDER BY id DESC
        """, conn)
    except Exception:
        # Fallback if table doesn't exist
        df = pd.DataFrame(columns=['Date', 'Time', 'Exercise', 'Reps', 'Duration (s)', 'Avg FPS'])
    conn.close()
    return df

def clear_workout_history():
    """Clears all records in the workout history table."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM workout_history")
    conn.commit()
    conn.close()
