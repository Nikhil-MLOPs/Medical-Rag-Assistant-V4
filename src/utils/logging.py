import logging
from pathlib import Path

# Providing path of log directory named logs
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok = True)

# Function to create logging
def setup_logging(name: str) -> logging.logger:
    logger = logging.get_logger(name) # get_logger -> creates a logger.
    logger.set_level(logging.INFO) # minimum level this logger will process is INFO

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s") # decides how each log line will look

    # Handler decides where the log goes
    console_handler = logging.StreamHandler() # Console handler so that we can see logs in the terminal
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_DIR / "app.log") # File handler so that logs are written in a file named logs/app.log
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False # If we didn’t set propagate = False, our log messages would appear twice (once from our handlers, and once from the root logger’s handler)

    return logger