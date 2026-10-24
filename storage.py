import logging
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from fastapi import HTTPException

from schemas import PredictionResponse


logger = logging.getLogger(__name__)

DATABASE_PATH = Path(__file__).resolve().parent / "agro_scoring.db"


@asynccontextmanager
async def get_db():
    async with aiosqlite.connect(
        str(DATABASE_PATH),
        timeout=5.0,
    ) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def init_db() -> None:
    async with get_db() as db:
        async with db.execute("PRAGMA journal_mode=WAL;"):
            pass

        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                farm_id TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                model_version TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_request_id
            ON predictions(request_id);
            """
        )

        await db.commit()

    logger.info("SQLite initialized | path=%s", DATABASE_PATH)


async def save_prediction(
    result: PredictionResponse,
    source: str = "single",
) -> None:
    try:
        async with get_db() as db:
            await db.execute(
                """
                INSERT INTO predictions (
                    request_id,
                    farm_id,
                    risk_score,
                    risk_level,
                    recommendation,
                    model_version,
                    source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    result.request_id,
                    result.farm_id,
                    result.risk_score,
                    result.risk_level,
                    result.recommendation,
                    result.model_version,
                    source,
                ),
            )

            await db.commit()

        logger.info(
            "Prediction committed | request_id=%s | source=%s",
            result.request_id,
            source,
        )

    except aiosqlite.Error as exc:
        logger.exception(
            "Database write failed | request_id=%s",
            result.request_id,
        )

        raise HTTPException(
            status_code=503,
            detail="Prediction storage is unavailable",
        ) from exc


async def list_predictions(
    limit: int,
    risk_level: str | None = None,
) -> list[dict]:
    try:
        async with get_db() as db:
            if risk_level is None:
                async with db.execute(
                    """
                    SELECT *
                    FROM predictions
                    ORDER BY id DESC
                    LIMIT ?;
                    """,
                    (limit,),
                ) as cursor:
                    rows = await cursor.fetchall()

            else:
                async with db.execute(
                    """
                    SELECT *
                    FROM predictions
                    WHERE risk_level = ?
                    ORDER BY id DESC
                    LIMIT ?;
                    """,
                    (risk_level, limit),
                ) as cursor:
                    rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    except aiosqlite.Error as exc:
        logger.exception("Database history read failed")

        raise HTTPException(
            status_code=503,
            detail="Prediction storage is unavailable",
        ) from exc


async def find_predictions(
    request_id: str,
) -> list[dict]:
    try:
        async with get_db() as db:
            async with db.execute(
                """
                SELECT *
                FROM predictions
                WHERE request_id = ?
                ORDER BY id ASC;
                """,
                (request_id,),
            ) as cursor:
                rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    except aiosqlite.Error as exc:
        logger.exception(
            "Database lookup failed | lookup_request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=503,
            detail="Prediction storage is unavailable",
        ) from exc
        
async def get_weekday_predictions(
    weekday_number: str,
    page: int,
    per_page: int,
) -> tuple[list[dict], int]:
    offset = (page - 1) * per_page

    try:
        async with get_db() as db:
            await db.execute("BEGIN;")

            async with db.execute(
                """
                SELECT COUNT(*)
                FROM predictions
                WHERE strftime('%w', created_at) = ?;
                """,
                (weekday_number,),
            ) as cursor:
                count_row = await cursor.fetchone()
                total_items = count_row[0]

            async with db.execute(
                """
                SELECT *
                FROM predictions
                WHERE strftime('%w', created_at) = ?
                ORDER BY id DESC
                LIMIT ? OFFSET ?;
                """,
                (weekday_number, per_page, offset),
            ) as cursor:
                rows = await cursor.fetchall()

            await db.commit()

        return [dict(row) for row in rows], total_items

    except aiosqlite.Error as exc:
        logger.exception(
            "Weekday pagination failed | weekday_number=%s",
            weekday_number,
        )

        raise HTTPException(
            status_code=503,
            detail="Prediction storage is unavailable",
        ) from exc