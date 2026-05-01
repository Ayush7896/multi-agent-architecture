# """"
# YOUR persistence layer. Completely separate from LangGraph internals.

# Concept map:
#   create_tables()      → runs once at app startup, creates YOUR tables
#   create_chat()        → starts a new conversation session
#   save_message()       → called by FastAPI before and after workflow.invoke()
#   load_chat_history()  → called by agents in Phase 2 (Memory injection)

# WHY psycopg2 and not SQLAlchemy?
#   psycopg2 = raw SQL driver. You see exactly what query runs.
#   SQLAlchemy = ORM that hides the SQL. Great for production, bad for learning.
#   We start with raw SQL so you understand what's happening, then you can
#   swap to SQLAlchemy or Tortoise ORM later.
# """

import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONNECTION CONFIG
# Reads from .env file. Never hardcode credentials.
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "travel_planner"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

def get_connection():
    """  
    Creates a fresh DB connection each call.

    WHY not a single global connection?
      psycopg2 connections are NOT thread-safe. FastAPI can handle
      multiple requests simultaneously, so each request needs its own
      connection. In production you'd use a connection pool (psycopg2.pool
      or SQLAlchemy pool). For now, one connection per call is clear and safe.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ DB connected")
        return conn
    except Exception as e:
        print("❌ DB connection failed:", e)
        raise

# ─────────────────────────────────────────────
# TABLE CREATION
# This is YOUR responsibility — LangGraph does NOT touch these tables.
# Called once at FastAPI startup.
# ─────────────────────────────────────────────

def create_tables():
    """ 
    Creates the chats and messages tables IF they don't exist.

    IF NOT EXISTS = safe to call every startup. If tables already
    exist, Postgres skips creation silently. No data loss.

    WHY gen_random_uuid()?
      Integer auto-increment IDs are sequential and guessable.
      UUIDs are random — a user cannot guess another user's chat_id
      by adding 1. This matters for security even in simple apps.
      Requires the pgcrypto extension (enabled by default in most
      Postgres installations).  
    WHY TIMESTAMPTZ not TIMESTAMP?
      TIMESTAMPTZ stores timezone-aware datetimes.
      TIMESTAMP stores naive datetimes (no timezone).
      Always use TIMESTAMPTZ — naive datetimes cause silent bugs
      when your server and DB are in different timezones.

    WHY ON DELETE CASCADE on messages?
      If you delete a chat row, all its messages are automatically
      deleted too. Without CASCADE, Postgres would refuse to delete
      a chat that still has messages (foreign key violation).
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Enable pgcrypto for gen_random_uuid()
        # This is idempotent — safe to run repeatedly
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

        # ── chats table ──────────────────────────────────────
        # One row per conversation session.
        # A user sends chat_id in their request to continue a session.
        # A missing chat_id = start a new chat.

        cursor.execute(""" 
        CREATE TABLE IF NOT EXISTS chats(
                       id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                       created_at TIMESTAMPTZ DEFAULT NOW()
                       )
        """)
        # ── messages table ────────────────────────────────────
        # One row per message (user or AI).
        # role is either 'user' or 'ai' — mirrors LangChain's
        # HumanMessage / AIMessage naming convention.

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
                chat_id    UUID        NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role       VARCHAR(10) NOT NULL CHECK (role IN ('user', 'ai')),
                content    TEXT        NOT NULL,
                timestamp  TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        # Index on chat_id so load_chat_history() is fast even with millions of rows.
        # Without this index, Postgres scans EVERY message row to find ones
        # matching a chat_id. With it, it jumps directly to the right rows.
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_chat_id
            ON messages(chat_id);
        """)
        conn.commit()
        print("[DB] Tables created (or already exist) - ready")

    except Exception as e:
        conn.rollback()
        print(f"[DB] Table creation failed: {e}")

    finally:
        # Always close cursor and connection — even if an error occurred.
        # 'finally' runs regardless of success or exception.
        cursor.close()
        conn.close()

# ─────────────────────────────────────────────
# CRUD FUNCTIONS
# Called by FastAPI (main.py) — not by LangGraph agents directly.
# In Phase 2, load_chat_history() will also be called from inside agents.
# ─────────────────────────────────────────────

def create_chat() -> str:
    """ 
    Creates a new chat session row. Returns the new chat_id as a string.

    WHY return a string and not UUID object?
      FastAPI serializes responses to JSON. UUID objects need explicit
      conversion. str() makes it immediately JSON-safe everywhere.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # RETURNING id = INSERT and immediately get the generated UUID back
        # without a second SELECT query.
        cursor.execute("INSERT INTO chats DEFAULT VALUES RETURNING id;")
        chat_id = str(cursor.fetchone()[0])
        conn.commit()
        return chat_id
    except Exception as e:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def save_message(chat_id: str,role: str, content: str ) -> None:
    """
    Saves one message to the messages table.

    Called TWICE per request in main.py:
      1. Before workflow.invoke()  → saves the user's message
      2. After  workflow.invoke()  → saves the AI's final_answer

    WHY %s placeholders and not f-strings?
      NEVER use f-strings or string concatenation for SQL values.
      f"... WHERE id = '{user_input}'" is SQL injection — a user can
      send '; DROP TABLE messages; --' as their input and destroy your DB.
      %s = parameterized query. psycopg2 escapes the value safely.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (%s, %s, %s);",
            (chat_id, role, content)  # tuple = positional params for %s
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def load_chat_history(chat_id: str, limit: int = 20) -> list[dict]:
    """  
    Loads the last `limit` messages for a chat, oldest first.

    Returns:
        [{"role": "user", "content": "...", "timestamp": ...}, ...]

    WHY RealDictCursor?
      Normal cursor returns tuples: (role, content, timestamp)
      RealDictCursor returns dicts: {"role": ..., "content": ...}
      Dicts are safer — you access by key not position, so adding
      columns to the table doesn't silently break your code.

    WHY limit?
      In Phase 2 you'll inject this history into the LLM prompt.
      LLMs have token limits. Injecting 1000 messages = expensive
      and likely exceeds the context window. Limit keeps it sane.

    WHY subquery with ORDER BY DESC + outer ORDER BY ASC?
      DESC LIMIT 20 = get the 20 MOST RECENT messages.
      Outer ASC = re-sort them chronologically oldest-to-newest.
      This ensures the LLM reads the conversation in the right order
    
    """    
    conn = get_connection()
    # 
    cursor = conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
    try:
        cursor.execute("""
            SELECT role, content, timestamp
            FROM (
                SELECT role, content, timestamp
                FROM messages
                WHERE chat_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            ) recent
            ORDER BY timestamp ASC;
        """, (chat_id, limit))
        rows = cursor.fetchall()
        # Convert RealDICTRow objects to plain python Dicts
        return [dict(row) for row in rows]
    finally:
        cursor.close()
        conn.close()