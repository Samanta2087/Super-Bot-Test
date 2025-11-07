"""
Configuration file for Super Bot
Keep this file LOCAL - do not upload to GitHub!
"""

import os
import json

# Telegram Bot Token
BOT_TOKEN = "7881711086:AAHE1d3k-a0O_w_T97rvPT-xXaW-XOdlhMI"

# Google Drive Credentials Path
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"

# Admin User ID
ADMIN_USER_ID = 7492936203

# Database
DATABASE_FILE = "bot_database.db"

# Premium System
FREE_UPLOAD_LIMIT_MB = 50
PREMIUM_UPLOAD_LIMIT_MB = 150

# Download Settings
DEFAULT_VIDEO_QUALITY = "720p"
AVAILABLE_QUALITIES = ["360p", "480p", "720p", "1080p"]

# Payment Settings
PAYMENT_UPI_ID = "samantapatra2087@ybl"
PAYMENT_AMOUNT = 100  # in rupees

def get_google_credentials():
    """Load Google credentials from file"""
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, 'r') as f:
            return json.load(f)
    return None

def get_google_token():
    """Load Google token from file"""
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            return json.load(f)
    return None
