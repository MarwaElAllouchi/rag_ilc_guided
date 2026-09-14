from app.core.logging_config import setup_logging
from batch.storage.pgvector_store import create_tables


setup_logging("INFO")

create_tables()
print("Tables OK")