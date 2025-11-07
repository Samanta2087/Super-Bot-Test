# Super Bot

A powerful Telegram bot with multiple features including video download, file conversion, Google Drive integration, and more.

## Features
- 📥 Video Downloader (YouTube, Instagram, etc.)
- 🖼️ Photos to PDF Converter
- 📁 Google Drive Integration
- 💾 File Storage & Management
- 👑 Premium System with Payment Integration
- 🎨 Multiple Quality Options
- ⚡ Fast Async Processing

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/Samanta2087/Super-Bot-Test.git
cd Super-Bot-Test
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example file and edit with your values:
```bash
copy .env.example .env
```

Then edit `.env` file:

#### A. Telegram Bot Token
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Create a new bot and get your token
3. Put the token in `.env` → `BOT_TOKEN=your_token_here`

#### B. Admin User ID
1. Get your Telegram User ID (message [@userinfobot](https://t.me/userinfobot))
2. Put your ID in `.env` → `ADMIN_USER_ID=your_id_here`

#### C. Google Drive API Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable Google Drive API
4. Create OAuth 2.0 credentials
5. Download credentials and save as `credentials.json` in the bot folder
6. Copy the client ID, secret, and project ID to `.env` file
7. Run the bot once to generate `token.json` (it will open browser for authorization)

### 4. Run the Bot
```bash
python super.py
```

## File Structure
```
Super-Bot-Test/
├── super.py              # Main bot code
├── .env                  # Your credentials (create from .env.example)
├── .env.example          # Example configuration file
├── requirements.txt      # Python dependencies
├── credentials.json      # Google Drive credentials (create this)
├── token.json           # Auto-generated after first run
└── bot_database.db      # Auto-generated database
```

## Important Notes
⚠️ **Never share these files publicly:**
- `.env` - Contains all your sensitive credentials
- `credentials.json` - Contains Google API secrets
- `token.json` - Contains access tokens
- `bot_database.db` - Contains user data

✅ **Safe to share:**
- `.env.example` - Template without real credentials
- `super.py` - Main code
- `requirements.txt` - Dependencies
- `README.md` - Documentation

## Features Configuration
All settings are in `.env` file:
- `BOT_TOKEN` - Your Telegram bot token
- `ADMIN_USER_ID` - Your Telegram user ID
- `PAYMENT_UPI_ID` - UPI ID for payments
- `PAYMENT_AMOUNT` - Premium price in rupees
- `FREE_UPLOAD_LIMIT` - Free tier limit (MB)
- `PREMIUM_UPLOAD_LIMIT` - Premium tier limit (MB)

## Support
For issues or questions, contact the developer.

## License
Private project - All rights reserved
