import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler


_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
	"""Configure application logging once with console and rotating file handlers.

	Respects LOG_LEVEL env var; defaults to INFO.
	Logs to data/logs/app.log relative to project root if writable.
	"""
	global _CONFIGURED
	if _CONFIGURED:
		return

	log_level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
	log_level = getattr(logging, log_level_name, logging.INFO)

	root_logger = logging.getLogger()
	root_logger.setLevel(log_level)

	# Prevent duplicate handlers if Streamlit reloads
	if root_logger.handlers:
		_CONFIGURED = True
		return

	formatter = logging.Formatter(
		fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

	# Console handler
	console = logging.StreamHandler()
	console.setLevel(log_level)
	console.setFormatter(formatter)
	root_logger.addHandler(console)

	# File handler (best-effort)
	try:
		logs_dir = Path(os.getenv("LOG_DIR", Path("data") / "logs"))
		logs_dir.mkdir(parents=True, exist_ok=True)
		file_handler = RotatingFileHandler(str(logs_dir / "app.log"), maxBytes=2_000_000, backupCount=3, encoding="utf-8")
		file_handler.setLevel(log_level)
		file_handler.setFormatter(formatter)
		root_logger.addHandler(file_handler)
	except Exception:
		# If file logging fails, continue with console only
		pass

	_CONFIGURED = True


