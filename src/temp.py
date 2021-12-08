import os
import helpers as h


logger = h.Logger()
CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")


logger.info(CURRENT_DIR)
logger.info(PROJECT_DIR)
logger.info(OUTPUT_DIR)