import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the Flask app
from app import app

# For cPanel deployment, we need to expose the app
application = app

if __name__ == "__main__":
    app.run()
