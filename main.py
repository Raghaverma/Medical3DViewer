"""
Main entry point for the Medical 3D Viewer application.
"""

import sys
import logging
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from config import LOG_LEVEL, LOG_FILE

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def main():
    """Main application entry point."""
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
