"""Application context: wires settings, database, provider and task runner together."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.data.providers import DataProvider, get_provider
from app.database.repository import Repository
from app.database.session import init_db, make_engine, make_session_factory
from app.models.intervals import IntervalService
from app.services.tasks import TaskManager


@dataclass
class AppContext:
    settings: Settings
    session_factory: sessionmaker[Session]
    provider: DataProvider
    tasks: TaskManager
    intervals: IntervalService

    @contextmanager
    def repo(self) -> Iterator[Repository]:
        """Session scope: commits on success, rolls back on error."""
        session = self.session_factory()
        try:
            yield Repository(session)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def build_context(settings: Settings | None = None, provider: DataProvider | None = None,
                  intervals: IntervalService | None = None) -> AppContext:
    settings = settings or get_settings()
    engine = make_engine(settings.resolved_database_url)
    init_db(engine)
    return AppContext(settings, make_session_factory(engine), provider or get_provider(settings),
                      TaskManager(), intervals or IntervalService(settings))
