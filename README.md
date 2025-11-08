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

#### C. Google Drive API Setup (IMPORTANT!)

**On Linux/Remote Server:**

Since you can't download files through browser on a remote server, you need to create `credentials.json` manually:

```bash
cd ~/Super-Bot-Test

# Create credentials.json (use your own values from Google Cloud Console)
cat > credentials.json << 'EOF'
{
  "installed": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "project_id": "your-project-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "YOUR_CLIENT_SECRET",
    "redirect_uris": ["http://localhost"]
  }
}
EOF
```

**For token.json (OAuth authorization):**

Option 1: Generate on your local PC and upload
1. Run bot on your local PC first
2. It will open browser for Google authorization
3. This creates `token.json`
4. Upload to server: `scp token.json user@server:~/Super-Bot-Test/`

Option 2: The bot will show instructions if token.json is missing

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
