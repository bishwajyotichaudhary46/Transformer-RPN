import logging
import os
from datetime import datetime

# defining log file and directory
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d')}.log"
logs_path = os.path.join(log_dir, LOG_FILE)

# set logging level
log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()

# confure logging
logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=getattr(logging, log_level, logging.DEBUG),
)

# add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

