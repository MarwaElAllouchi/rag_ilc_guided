import logging

from app.core.logging_config import setup_logging


setup_logging("INFO")

logger = logging.getLogger(__name__)

logger.info("Test logging OK")
logger.warning("Test warning OK")