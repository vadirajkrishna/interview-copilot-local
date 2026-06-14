import aiosqlite

from config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    company TEXT,
    role TEXT,
    duration INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS exchanges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT DEFAULT '',
    framework_used TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange_id INTEGER NOT NULL,
    steps_covered_json TEXT NOT NULL,
    score INTEGER NOT NULL,
    feedback TEXT NOT NULL,
    FOREIGN KEY (exchange_id) REFERENCES exchanges(id)
);

CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    labelled_json TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS patterns (
    framework TEXT PRIMARY KEY,
    times_shown INTEGER NOT NULL DEFAULT 0,
    avg_score REAL NOT NULL DEFAULT 0,
    most_missed_step TEXT DEFAULT ''
);
"""


async def init_db(db_path=DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA)
        await db.commit()

