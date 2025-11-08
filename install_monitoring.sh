#!/bin/bash
# Install Bot Monitoring System

echo "🔧 Installing Bot Monitoring System..."
echo "======================================="

# Install required packages
pip install --upgrade python-telegram-bot

# Initialize monitor database
python3 bot_monitor.py

# Setup cron job for daily reports (9 AM every day)
echo "📅 Setting up daily report schedule..."

# Create cron job
(crontab -l 2>/dev/null; echo "0 9 * * * cd $(pwd) && python3 bot_monitor.py") | crontab -

echo ""
echo "✅ Monitoring system installed!"
echo ""
echo "📊 What's installed:"
echo "  • bot_monitor.py - Core monitoring system"
echo "  • bot_monitor.db - Monitoring database"
echo "  • Daily reports at 9:00 AM to admin"
echo ""
echo "📝 Next steps:"
echo "  1. Integrate monitoring into super.py (see MONITORING_INTEGRATION.py)"
echo "  2. Test manual report: python3 bot_monitor.py"
echo "  3. Monitor runs automatically with your bot!"
echo ""
echo "🎯 Features:"
echo "  • Tracks all user activities"
echo "  • Monitors success/failure rates"
echo "  • Logs all errors with details"
echo "  • Sends daily summary reports"
echo "  • Shows trending data"
echo "  • Identifies top users"
echo ""
