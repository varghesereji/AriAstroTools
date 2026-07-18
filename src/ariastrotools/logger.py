import logging
import sys

logger = logging.getLogger("AriAstroTools")

# If the application hasn't configured logging, log to the console.
if not logging.getLogger().hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
