import logging
import sys

logger = logging.getLogger("AriAstroTools")
logger.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

# File handler
log_filename = "Ariastro_logs.log"
fh = logging.FileHandler(log_filename)
fh.setLevel(logging.INFO)


# Formatter
formatter = logging.Formatter(
    "%(asctime) s -%(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(ch)
    logger.addHandler(fh)
