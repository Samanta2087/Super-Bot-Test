# 📊 Bot Activity Monitoring System

Complete A-to-Z tracking of your Telegram bot with daily reports sent directly to you!

## 🎯 What It Tracks

### User Activity
- Total users (daily/weekly/monthly)
- New users
- Most active users
- User engagement rates

### Bot Performance
- Total requests
- Success/failure rates
- Average processing time
- Data transferred (uploads/downloads)

### Feature Usage
- Which features are most popular
- Usage trends over time
- Feature-specific statistics

### Error Monitoring
- All errors with full details
- Error frequency and patterns
- Stack traces for debugging
- User-specific issues

### Daily Reports Include
✅ User statistics (total, new, active)
✅ Activity breakdown (uploads, downloads, requests)
✅ Success rate percentage
✅ Most used features (top 5)
✅ Top active users (top 5)
✅ Error summary (if any)
✅ Trend analysis (week-over-week growth)
✅ Data usage in MB

## 📦 Installation

### Quick Install (Linux Server)

```bash
cd ~/Super-Bot-Test

# Make scripts executable
chmod +x install_monitoring.sh

# Run installation
./install_monitoring.sh
```

### Manual Install

```bash
# 1. Ensure bot_monitor.py is in your bot directory
# 2. Initialize database
python3 bot_monitor.py

# 3. Setup daily report (9 AM)
(crontab -l 2>/dev/null; echo "0 9 * * * cd $(pwd) && python3 bot_monitor.py") | crontab -
```

## 🔧 Integration with Your Bot

### Automatic Integration (Recommended)

```bash
python3 integrate_monitoring.py
```

This will:
- ✅ Add monitoring imports
- ✅ Initialize monitor system
- ✅ Add daily report scheduler
- ✅ Create backup of original file

### Manual Integration

Add to your `super.py`:

```python
# 1. Import (after dotenv)
from bot_monitor import BotMonitor
import time as time_module

# 2. Initialize (after logger)
monitor = BotMonitor()

# 3. Wrap handlers
async def your_command(update, context):
    start_time = time_module.time()
    user_id = update.effective_user.id
    username = update.effective_user.username
    
    try:
        # Your existing code here
        
        # Log success
        monitor.log_activity(
            user_id=user_id,
            username=username,
            action='your_command',
            feature='Feature Name',
            success=True,
            processing_time=time_module.time() - start_time
        )
        monitor.update_feature_usage('Feature Name')
        
    except Exception as e:
        # Log error
        monitor.log_error(
            error_type=type(e).__name__,
            error_message=str(e),
            user_id=user_id,
            feature='Feature Name'
        )
        raise

# 4. Add daily report job (in main function)
async def send_daily_report():
    await monitor.send_daily_report_to_admin()

scheduler.add_job(
    send_daily_report,
    CronTrigger(hour=9, minute=0),
    id='daily_report'
)
```

## 📱 Daily Report Example

```
📊 DAILY BOT ACTIVITY REPORT
📅 Date: 2025-11-07
==================================================

👥 USER STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Users: 45
• New Users: 8
• Total Requests: 234
• Success Rate: 96.5%

📊 ACTIVITY BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━
• ✅ Successful: 226
• ❌ Failed: 8
• ⬆️ Uploads: 89
• ⬇️ Downloads: 145
• 💾 Data Processed: 2,345.67 MB
• ⚡ Avg Processing Time: 3.45s

🔥 MOST USED FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Video Download: 145 times
2. File Upload: 89 times
3. Photo to PDF: 34 times
4. Premium Purchase: 12 times
5. Compress Video: 8 times

👑 TOP ACTIVE USERS
━━━━━━━━━━━━━━━━━━━━━━━━━━
1. @username1: 45 actions
2. @username2: 32 actions
3. @username3: 28 actions

⚠️ ERROR SUMMARY (8 total)
━━━━━━━━━━━━━━━━━━━━━━━━━━
• NetworkError: Connection timeout... (3x)
• FileNotFoundError: File not accessible... (2x)

==================================================
📈 TREND ANALYSIS
• User Growth (vs last week): +15.5%
• Request Growth: +22.3%

🤖 Generated at: 2025-11-08 09:00:00
```

## 🛠️ Manual Report

Generate report anytime:

```bash
python3 bot_monitor.py
```

## 📊 Database Structure

### activity_log
- Tracks every user action
- Stores timing, success/failure
- Records file sizes

### error_log
- All errors with full context
- Stack traces for debugging
- User and feature information

### feature_usage
- Daily usage counts per feature
- Trend analysis data

### daily_stats
- Pre-calculated daily statistics
- Fast report generation

## 🔍 Advanced Usage

### Custom Reports

```python
from bot_monitor import BotMonitor

monitor = BotMonitor()

# Get specific date stats
stats = monitor.get_daily_stats('2025-11-07')

# Get error summary
errors = monitor.get_error_summary('2025-11-07')

# Get top users
top_users = monitor.get_top_users(date='2025-11-07', limit=10)
```

### Manual Logging

```python
# Log custom activity
monitor.log_activity(
    user_id=12345,
    username='user123',
    action='custom_action',
    feature='My Feature',
    file_size=1024000,  # in bytes
    success=True,
    processing_time=2.5  # seconds
)

# Log custom error
monitor.log_error(
    error_type='CustomError',
    error_message='Something went wrong',
    user_id=12345,
    feature='My Feature',
    stack_trace='Full traceback here'
)

# Update feature usage
monitor.update_feature_usage('My Feature')
```

## ⏰ Report Schedule

Default: **9:00 AM daily**

Change schedule in cron:
```bash
crontab -e

# Modify to your preferred time
0 21 * * * cd /path/to/bot && python3 bot_monitor.py  # 9 PM
```

Or in super.py scheduler:
```python
scheduler.add_job(
    send_daily_report,
    CronTrigger(hour=21, minute=0),  # 9 PM
    id='daily_report'
)
```

## 🔐 Security

- Database is local only
- Reports sent only to ADMIN_USER_ID
- No external APIs used
- User data stays on your server

## 📁 Files Created

- `bot_monitor.db` - Monitoring database
- `super.py.backup.YYYYMMDD_HHMMSS` - Backup files

## 🐛 Troubleshooting

**Report not received?**
- Check ADMIN_USER_ID in .env
- Verify BOT_TOKEN is correct
- Check bot_monitor.db exists
- Run manual test: `python3 bot_monitor.py`

**Integration errors?**
- Restore from backup: `cp super.py.backup.* super.py`
- Check Python version >= 3.7
- Ensure all imports are correct

## 📞 Support

Check the logs:
```bash
tail -f bot_monitor.log  # If logging to file
```

Manual database check:
```bash
sqlite3 bot_monitor.db "SELECT COUNT(*) FROM activity_log;"
```

## 🎉 Benefits

✅ Know exactly what users are doing
✅ Identify and fix errors quickly
✅ See which features are popular
✅ Track bot growth over time
✅ Get insights for improvements
✅ Monitor performance issues
✅ Daily automated reports to your Telegram

---

**Made with ❤️ for Super Bot**
Tracks everything A to Z! 📊
