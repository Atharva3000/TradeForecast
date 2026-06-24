import sqlite3
import os

# Check if running on Vercel or in a read-only environment.
# Vercel filesystem is read-only except for /tmp.
if os.environ.get("VERCEL") == "1" or os.environ.get("VERCEL") is not None:
    DB_PATH = "/tmp/users.db"
else:
    # Check if we have write access to the project root directory
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _local_db_path = os.path.join(_project_root, "users.db")
    try:
        # Test if we can write to the directory
        _test_path = os.path.join(_project_root, ".db_write_test")
        with open(_test_path, "w") as f:
            f.write("test")
        os.remove(_test_path)
        DB_PATH = _local_db_path
    except (IOError, OSError, PermissionError):
        DB_PATH = "/tmp/users.db"

def get_db_connection():
    """Get a connection to the SQLite database with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        name TEXT,
        email TEXT,
        trading_experience TEXT,
        investment_capital REAL,
        country TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    conn.commit()
    conn.close()
