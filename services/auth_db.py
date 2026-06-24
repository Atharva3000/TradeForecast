import hashlib
import os
import secrets
from services.db import get_db_connection

def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with a random salt."""
    salt = os.urandom(16).hex()
    iterations = 100000
    hash_bytes = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations
    )
    hash_hex = hash_bytes.hex()
    return f"pbkdf2:sha256:{iterations}${salt}${hash_hex}"

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its PBKDF2 hash."""
    try:
        if not hashed.startswith("pbkdf2:sha256:"):
            return False
        parts = hashed.split("$")
        if len(parts) != 3:
            return False
        meta, salt, hash_hex = parts
        iterations = int(meta.split(":")[-1])
        
        test_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations
        )
        return secrets.compare_digest(test_hash.hex(), hash_hex)
    except Exception:
        return False

def register_user(username: str, password: str, name: str = None, email: str = None,
                  trading_experience: str = None, investment_capital: float = None,
                  country: str = None):
    """
    Registers a new user in the database.
    Returns the user data dict if successful, raises ValueError if username exists.
    """
    username = username.strip().lower()
    hashed = hash_password(password)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            INSERT INTO users (username, password_hash, name, email, trading_experience, investment_capital, country)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (username, hashed, name, email, trading_experience, investment_capital, country)
        )
        conn.commit()
        
        # Get the newly created user
        cursor.execute(
            "SELECT id, username, name, email, trading_experience, investment_capital, country FROM users WHERE username = ?",
            (username,)
        )
        user = cursor.fetchone()
        return dict(user)
    except sqlite3.IntegrityError:
        raise ValueError(f"Username '{username}' is already taken.")
    finally:
        conn.close()

# Import sqlite3 here to catch the IntegrityError specifically inside register_user
import sqlite3

def authenticate_user(username: str, password: str):
    """
    Authenticates a user against username and password.
    Returns the user record dict if successful, None otherwise.
    """
    username = username.strip().lower()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, username, password_hash, name, email, trading_experience, investment_capital, country FROM users WHERE username = ?",
        (username,)
    )
    user = cursor.fetchone()
    conn.close()
    
    if user and verify_password(password, user["password_hash"]):
        user_dict = dict(user)
        # remove password_hash for safety in returned values
        user_dict.pop("password_hash", None)
        return user_dict
    
    return None

def update_user_profile(username: str, name: str, email: str,
                        trading_experience: str, investment_capital: float,
                        country: str):
    """
    Updates the profile information for a user.
    Returns the updated user record dict.
    """
    username = username.strip().lower()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        UPDATE users
        SET name = ?, email = ?, trading_experience = ?, investment_capital = ?, country = ?
        WHERE username = ?
        """,
        (name, email, trading_experience, investment_capital, country, username)
    )
    conn.commit()
    
    cursor.execute(
        "SELECT id, username, name, email, trading_experience, investment_capital, country FROM users WHERE username = ?",
        (username,)
    )
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return dict(user)
    return None
