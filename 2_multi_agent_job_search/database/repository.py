"""
database/repository.py
───────────────────────
Generic + model-specific repository classes.
All DB access goes through these — never raw queries in agent/tool code.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Optional, Sequence, Type, TypeVar

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    Application,
    ApplicationStatus,
    Job,
    MatchScore,
    Notification,
    Resume,
    SearchHistory,
)

T = TypeVar("T")


class BaseRepository(Generic[T]):
    def __init__(self, session: AsyncSession, model: Type[T]) -> None:
        self.session = session
        self.model = model

    async def get(self, id: str) -> Optional[T]:
        return await self.session.get(self.model, id)

    async def list(self, limit: int = 100, offset: int = 0) -> Sequence[T]:
        result = await self.session.execute(
            select(self.model).offset(offset).limit(limit)
        )
        return result.scalars().all()

    async def create(self, obj: T) -> T:
        self.session.add(obj)
        await self.session.flush()
        await self.session.refresh(obj)
        return obj

    async def delete(self, obj: T) -> None:
        await self.session.delete(obj)
        await self.session.flush()


class JobRepository(BaseRepository[Job]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Job)

    async def get_by_external_id(self, external_id: str) -> Optional[Job]:
        result = await self.session.execute(
            select(Job).where(Job.external_id == external_id)
        )
        return result.scalar_one_or_none()

    async def search(
        self,
        keywords: Optional[list[str]] = None,
        location: Optional[str] = None,
        is_remote: Optional[bool] = None,
        min_salary: Optional[float] = None,
        source: Optional[str] = None,
        limit: int = 50,
    ) -> Sequence[Job]:
        q = select(Job).where(Job.is_active == True)
        if location:
            q = q.where(Job.location.ilike(f"%{location}%"))
        if is_remote is not None:
            q = q.where(Job.is_remote == is_remote)
        if min_salary:
            q = q.where(Job.salary_min >= min_salary)
        if source:
            q = q.where(Job.source == source)
        q = q.order_by(Job.created_at.desc()).limit(limit)
        result = await self.session.execute(q)
        return result.scalars().all()

    async def count_by_source(self) -> dict[str, int]:
        result = await self.session.execute(
            select(Job.source, func.count(Job.id)).group_by(Job.source)
        )
        return {row[0]: row[1] for row in result.all()}

    async def deduplicate(self) -> int:
        """Mark duplicate jobs as inactive. Returns count of dupes removed."""
        # Simple dedup: same title + company within 7 days
        # Full implementation would use cosine similarity on embeddings
        return 0  # placeholder — see VectorStore for semantic dedup


class ResumeRepository(BaseRepository[Resume]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Resume)

    async def get_active(self) -> Optional[Resume]:
        result = await self.session.execute(
            select(Resume).where(Resume.is_active == True).order_by(Resume.created_at.desc())
        )
        return result.scalar_one_or_none()


class ApplicationRepository(BaseRepository[Application]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Application)

    async def get_by_status(self, status: ApplicationStatus) -> Sequence[Application]:
        result = await self.session.execute(
            select(Application).where(Application.status == status)
        )
        return result.scalars().all()

    async def status_counts(self) -> dict[str, int]:
        result = await self.session.execute(
            select(Application.status, func.count(Application.id)).group_by(Application.status)
        )
        return {row[0]: row[1] for row in result.all()}

    async def update_status(self, app_id: str, status: ApplicationStatus, **kwargs: Any) -> None:
        await self.session.execute(
            update(Application)
            .where(Application.id == app_id)
            .values(status=status, updated_at=datetime.utcnow(), **kwargs)
        )


class MatchScoreRepository(BaseRepository[MatchScore]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, MatchScore)

    async def top_matches(self, resume_id: str, limit: int = 10) -> Sequence[MatchScore]:
        result = await self.session.execute(
            select(MatchScore)
            .where(MatchScore.resume_id == resume_id)
            .order_by(MatchScore.score.desc())
            .limit(limit)
        )
        return result.scalars().all()


class NotificationRepository(BaseRepository[Notification]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Notification)

    async def get_pending(self) -> Sequence[Notification]:
        result = await self.session.execute(
            select(Notification).where(Notification.is_sent == False)
        )
        return result.scalars().all()


class SearchHistoryRepository(BaseRepository[SearchHistory]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, SearchHistory)
