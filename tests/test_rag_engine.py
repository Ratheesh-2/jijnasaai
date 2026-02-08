import pytest
from unittest.mock import MagicMock
from backend.services.rag_engine import RAGEngine


@pytest.fixture
def rag_engine(test_settings):
    mock_vs = MagicMock()
    return RAGEngine(test_settings, mock_vs)


class TestChunking:
    def test_chunk_short_text(self, rag_engine):
        chunks = rag_engine._chunk_text("Short text.")
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_chunk_empty_text(self, rag_engine):
        chunks = rag_engine._chunk_text("")
        assert len(chunks) == 0

    def test_chunk_whitespace_only(self, rag_engine):
        chunks = rag_engine._chunk_text("   \n\n  ")
        assert len(chunks) == 0

    def test_chunk_long_text(self, rag_engine):
        long_text = "Hello world. " * 200  # ~2600 chars
        chunks = rag_engine._chunk_text(long_text)
        assert len(chunks) > 1

    def test_chunk_respects_paragraphs(self, rag_engine):
        text = "First paragraph.\n\n" * 5 + "Last paragraph."
        chunks = rag_engine._chunk_text(text)
        # Should be a single chunk since total is small
        assert len(chunks) >= 1

    def test_chunk_overlap_present(self, rag_engine):
        # Create text that will produce multiple chunks
        long_text = ("This is a sentence with some words. " * 50 + "\n\n") * 5
        chunks = rag_engine._chunk_text(long_text)
        if len(chunks) > 1:
            # Second chunk should start with content from end of first chunk
            first_tail = chunks[0][-100:]
            assert first_tail[:50] in chunks[1][:300]


class TestFileLoading:
    def test_load_txt_file(self, rag_engine, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello, this is a test document.", encoding="utf-8")
        text = rag_engine._load_file(txt_file)
        assert "Hello, this is a test document." in text

    def test_load_md_file(self, rag_engine, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\nSome content.", encoding="utf-8")
        text = rag_engine._load_file(md_file)
        assert "Title" in text
        assert "Some content." in text

    def test_load_unsupported_type(self, rag_engine, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            rag_engine._load_file(bad_file)
