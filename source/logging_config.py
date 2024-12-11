import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),  # Logs to a file
        logging.StreamHandler()         # Logs to console
    ]
)

# Optional: Create a logger instance for specific use
logger = logging.getLogger("AppLogger")
