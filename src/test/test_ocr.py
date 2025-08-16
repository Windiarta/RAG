from pathlib import Path

from src.modules.ocr import extract_text


def test_extract_text_txt(tmp_path: Path):
	file = tmp_path / "a.txt"
	file.write_text("hello\nworld", encoding="utf-8")
	text = extract_text(str(file))
	assert "hello" in text and "world" in text


