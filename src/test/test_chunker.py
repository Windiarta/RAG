from src.modules.chunker import chunk_text


def test_chunker_basic():
	text = "A" * 2500
	chunks = chunk_text(text, chunk_size=1000, overlap=200)
	assert len(chunks) == 3
	assert len(chunks[0]) <= 1000
	assert len(chunks[1]) <= 1000
	assert len(chunks[2]) <= 1000


def test_chunker_no_text():
	assert chunk_text("") == []


