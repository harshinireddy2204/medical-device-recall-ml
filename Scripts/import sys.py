import sys
import logging
import importlib.util

# Initialize logging early
logging.basicConfig(level=logging.INFO, format="%(asctime)s [INFO] %(message)s")

# Log environment info
logging.info(f"⚙️ Python executable: {sys.executable}")
logging.info(f"⚙️ Python version: {sys.version}")

# Check if ijson is installed and available
if importlib.util.find_spec("ijson") is not None:
    import ijson
    logging.info("✅ ijson module detected and successfully imported.")
else:
    logging.error("❌ ijson module NOT found in this Python environment. Please install it using: pip install ijson")
    sys.exit(1)
