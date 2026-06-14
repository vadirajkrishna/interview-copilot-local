import json
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from config import DB_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_session(
    company: str = "",
    role: str = "",
    duration: int = 0,
    db_path=DB_PATH,
) -> int:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "INSERT INTO sessions (date, company, role, duration) VALUES (?, ?, ?, ?)",
            (utc_now(), company, role, duration),
        )
        await db.commit()
        return int(cursor.lastrowid)


async def add_exchange(
    session_id: int,
    question: str,
    answer: str,
    framework_used: str,
    db_path=DB_PATH,
) -> int:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            INSERT INTO exchanges (session_id, question, answer, framework_used, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, question, answer, framework_used, utc_now()),
        )
        await db.execute(
            """
            INSERT INTO patterns (framework, times_shown)
            VALUES (?, 1)
            ON CONFLICT(framework) DO UPDATE SET times_shown = times_shown + 1
            """,
            (framework_used,),
        )
        await db.commit()
        return int(cursor.lastrowid)


async def add_transcript(
    session_id: int,
    raw_text: str,
    labelled: dict[str, Any],
    db_path=DB_PATH,
) -> int:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "INSERT INTO transcripts (session_id, raw_text, labelled_json) VALUES (?, ?, ?)",
            (session_id, raw_text, json.dumps(labelled)),
        )
        await db.commit()
        return int(cursor.lastrowid)


async def add_evaluation(
    exchange_id: int,
    steps_covered: list[bool],
    score: int,
    feedback: str,
    db_path=DB_PATH,
) -> int:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            INSERT INTO evaluations (exchange_id, steps_covered_json, score, feedback)
            VALUES (?, ?, ?, ?)
            """,
            (exchange_id, json.dumps(steps_covered), score, feedback),
        )
        await db.commit()
        return int(cursor.lastrowid)


async def update_exchange_answer(exchange_id: int, answer: str, db_path=DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE exchanges SET answer = ? WHERE id = ?",
            (answer, exchange_id),
        )
        await db.commit()


async def append_exchange_answer(exchange_id: int, addition: str, db_path=DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT answer FROM exchanges WHERE id = ?",
            (exchange_id,),
        )
        row = await cursor.fetchone()
        existing = row[0] if row else ""
        updated = f"{existing.strip()}\n{addition.strip()}".strip()
        await db.execute(
            "UPDATE exchanges SET answer = ? WHERE id = ?",
            (updated, exchange_id),
        )
        await db.commit()


async def clear_all_tables(db_path=DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(
            """
            DELETE FROM evaluations;
            DELETE FROM exchanges;
            DELETE FROM transcripts;
            DELETE FROM patterns;
            DELETE FROM sessions;
            DELETE FROM sqlite_sequence
            WHERE name IN ('evaluations', 'exchanges', 'transcripts', 'sessions');
            """
        )
        await db.commit()


async def list_exchanges(session_id: int, db_path=DB_PATH) -> list[dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT id, session_id, question, answer, framework_used, timestamp
            FROM exchanges
            WHERE session_id = ?
            ORDER BY id
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def list_evaluations(session_id: int, db_path=DB_PATH) -> list[dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT
                evaluations.id,
                evaluations.exchange_id,
                exchanges.session_id,
                sessions.date,
                sessions.company,
                sessions.role,
                exchanges.question,
                exchanges.answer,
                exchanges.framework_used,
                evaluations.steps_covered_json,
                evaluations.score,
                evaluations.feedback
            FROM evaluations
            JOIN exchanges ON evaluations.exchange_id = exchanges.id
            JOIN sessions ON exchanges.session_id = sessions.id
            WHERE exchanges.session_id = ?
            ORDER BY evaluations.id
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["steps_covered"] = json.loads(item.pop("steps_covered_json"))
            results.append(item)
        return results


async def list_all_evaluations(db_path=DB_PATH) -> list[dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT
                evaluations.id,
                evaluations.exchange_id,
                exchanges.session_id,
                sessions.date,
                sessions.company,
                sessions.role,
                exchanges.question,
                exchanges.answer,
                exchanges.framework_used,
                evaluations.steps_covered_json,
                evaluations.score,
                evaluations.feedback
            FROM evaluations
            JOIN exchanges ON evaluations.exchange_id = exchanges.id
            JOIN sessions ON exchanges.session_id = sessions.id
            ORDER BY sessions.id DESC, evaluations.id DESC
            """
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["steps_covered"] = json.loads(item.pop("steps_covered_json"))
            results.append(item)
        return results
