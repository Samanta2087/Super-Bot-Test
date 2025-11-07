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

### 3. Configure Credentials

#### A. Telegram Bot Token
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Create a new bot and get your token
3. Edit `config.py` and replace `BOT_TOKEN` with your token

#### B. Google Drive API Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable Google Drive API
4. Create OAuth 2.0 credentials
5. Download credentials and save as `credentials.json` in the bot folder
6. Run the bot once to generate `token.json` (it will open browser for authorization)

#### C. Admin Configuration
1. Get your Telegram User ID (message [@userinfobot](https://t.me/userinfobot))
2. Edit `config.py` and replace `ADMIN_USER_ID` with your ID

### 4. Run the Bot
```bash
python super.py
```

## File Structure
```
Super-Bot-Test/
├── super.py              # Main bot code
├── config.py             # Configuration (BOT_TOKEN, settings)
├── requirements.txt      # Python dependencies
├── credentials.json      # Google Drive credentials (create this)
├── token.json           # Auto-generated after first run
└── bot_database.db      # Auto-generated database
```

## Important Notes
⚠️ **Never share these files publicly:**
- `credentials.json` - Contains Google API secrets
- `token.json` - Contains access tokens
- `config.py` - Contains your bot token
- `bot_database.db` - Contains user data

## Features Configuration
Edit `config.py` to customize:
- Upload limits (Free: 50MB, Premium: 150MB)
- Video quality options
- Payment UPI ID
- Premium pricing

## Support
For issues or questions, contact the developer.

## License
Private project - All rights reserved
