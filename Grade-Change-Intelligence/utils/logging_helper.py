import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "recommendations_log.db"

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            transition TEXT,
            sim_time INTEGER,
            parameter TEXT,
            curr_val REAL,
            prop_val REAL,
            diff REAL,
            source TEXT,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_recommendations(transition, sim_time, actions_list):
    """
    actions_list: list of dicts from recommender
    Returns list of logged row IDs
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    ids = []
    for act in actions_list:
        cursor.execute("""
            INSERT INTO recommendations (transition, sim_time, parameter, curr_val, prop_val, diff, source, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Shown')
        """, (transition, sim_time, act["parameter"], act["curr_val"], act["prop_val"], act["diff"], act["source"]))
        ids.append(cursor.lastrowid)
    conn.commit()
    conn.close()
    return ids

def update_recommendations_status(ids, status):
    if not ids:
        return
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in ids)
    cursor.execute(f"""
        UPDATE recommendations
        SET status = ?
        WHERE id IN ({placeholders})
    """, [status] + list(ids))
    conn.commit()
    conn.close()

def get_acceptance_metrics():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total accepted
    cursor.execute("SELECT COUNT(*) FROM recommendations WHERE status = 'Accepted'")
    accepted = cursor.fetchone()[0]
    
    # Total rejected
    cursor.execute("SELECT COUNT(*) FROM recommendations WHERE status = 'Rejected'")
    rejected = cursor.fetchone()[0]
    
    # Total shown
    cursor.execute("SELECT COUNT(*) FROM recommendations")
    total_shown = cursor.fetchone()[0]
    
    conn.close()
    
    total_decided = accepted + rejected
    rate = (accepted / total_decided) if total_decided > 0 else 0.0
    return {
        "accepted": accepted,
        "rejected": rejected,
        "total_shown": total_shown,
        "acceptance_rate": rate
    }
