import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from backend.database import set_db_path, init_db


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def test_settings(temp_db_path, tmp_path):
    from backend.config import Settings
    return Settings(
        openai_api_key="sk-test-fake",
        anthropic_api_key="sk-ant-test-fake",
        google_api_key="fake-google-key",
        perplexity_api_key="pplx-test-fake",
        database_url=temp_db_path,
        chroma_persist_dir=str(tmp_path / "chroma"),

    )


@pytest_asyncio.fixture
async def initialized_db(temp_db_path):
    set_db_path(temp_db_path)
    await init_db()
    yield temp_db_path
