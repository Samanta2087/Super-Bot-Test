"""
Bot Activity Monitor - Tracks everything A to Z
Sends daily reports to admin about bot usage, errors, and statistics
"""

import sqlite3
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os
from telegram import Bot
from telegram.error import TelegramError

# Configuration
DATABASE_FILE = 'bot_database.db'
MONITOR_DB = 'bot_monitor.db'
ADMIN_ID = int(os.getenv('ADMIN_USER_ID', 0))
BOT_TOKEN = os.getenv('BOT_TOKEN')

class BotMonitor:
    def __init__(self):
        self.init_monitor_db()
        
    def init_monitor_db(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(MONITOR_DB)
        cursor = conn.cursor()
        
        # Activity tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                username TEXT,
                action TEXT NOT NULL,
                feature TEXT,
                file_size INTEGER,
                success INTEGER,
                error_message TEXT,
                processing_time REAL
            )
        ''')
        
        # Daily statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                total_users INTEGER,
                new_users INTEGER,
                total_requests INTEGER,
                successful_requests INTEGER,
                failed_requests INTEGER,
                total_uploads INTEGER,
                total_downloads INTEGER,
                total_data_mb REAL,
                premium_purchases INTEGER,
                errors_count INTEGER
            )
        ''')
        
        # Error log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                error_type TEXT,
                error_message TEXT,
                user_id INTEGER,
                feature TEXT,
                stack_trace TEXT
            )
        ''')
        
        # Feature usage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_usage (
                date TEXT NOT NULL,
                feature TEXT NOT NULL,
                usage_count INTEGER,
                PRIMARY KEY (date, feature)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Bot Monitor database initialized")
    
    def log_activity(self, user_id, username, action, feature=None, file_size=0, 
                     success=True, error_message=None, processing_time=0):
        """Log any bot activity"""
        conn = sqlite3.connect(MONITOR_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO activity_log 
            (timestamp, user_id, username, action, feature, file_size, success, error_message, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), user_id, username, action, feature, 
              file_size, 1 if success else 0, error_message, processing_time))
        
        conn.commit()
        conn.close()
    
    def log_error(self, error_type, error_message, user_id=None, feature=None, stack_trace=None):
        """Log errors"""
        conn = sqlite3.connect(MONITOR_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO error_log (timestamp, error_type, error_message, user_id, feature, stack_trace)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), error_type, error_message, user_id, feature, stack_trace))
        
        conn.commit()
        conn.close()
    
    def update_feature_usage(self, feature):
        """Track feature usage"""
        today = datetime.now().date().isoformat()
        conn = sqlite3.connect(MONITOR_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feature_usage (date, feature, usage_count)
            VALUES (?, ?, 1)
            ON CONFLICT(date, feature) 
            DO UPDATE SET usage_count = usage_count + 1
        ''', (today, feature))
        
        conn.commit()
        conn.close()
    
    def get_daily_stats(self, date=None):
        """Get statistics for a specific date"""
        if date is None:
            date = (datetime.now() - timedelta(days=1)).date().isoformat()
        
        conn = sqlite3.connect(MONITOR_DB)
        cursor = conn.cursor()
        
        # Get activity stats
        cursor.execute('''
            SELECT 
                COUNT(DISTINCT user_id) as total_users,
                COUNT(*) as total_requests,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_requests,
                SUM(CASE WHEN action LIKE '%upload%' THEN 1 ELSE 0 END) as total_uploads,
                SUM(CASE WHEN action LIKE '%download%' THEN 1 ELSE 0 END) as total_downloads,
                SUM(file_size) / 1024.0 / 1024.0 as total_data_mb,
                AVG(processing_time) as avg_processing_time
            FROM activity_log
            WHERE DATE(timestamp) = ?
        ''', (date,))
        
        stats = cursor.fetchone()
        
        # Get error count
        cursor.execute('''
            SELECT COUNT(*) FROM error_log
            WHERE DATE(timestamp) = ?
        ''', (date,))
        
        error_count = cursor.fetchone()[0]
        
        # Get feature usage
        cursor.execute('''
            SELECT feature, usage_count
            FROM feature_usage
            WHERE date = ?
            ORDER BY usage_count DESC
        ''', (date,))
        
        feature_stats = cursor.fetchall()
        
        # Get new users (first time users)
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) 
            FROM activity_log
            WHERE DATE(timestamp) = ?
            AND user_id NOT IN (
                SELECT DISTINCT user_id FROM activity_log
                WHERE DATE(timestamp) < ?
            )
        ''', (date, date))
        
        new_users = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'date': date,
            'total_users': stats[0] or 0,
            'new_users': new_users or 0,
            'total_requests': stats[1] or 0,
            'successful_requests': stats[2] or 0,
            'failed_requests': stats[3] or 0,
            'total_uploads': stats[4] or 0,
            'total_downloads': stats[5] or 0,
            'total_data_mb': round(stats[6] or 0, 2),
            'avg_processing_time': round(stats[7] or 0, 2),
            'errors_count': error_count,
            'feature_stats': feature_stats
        }
    
    def get_error_summary(self, date=None):
        """Get error summary for a date"""
        if date is None:
            date = (datetime.now() - timedelta(days=1)).date().isoformat()
        
        conn = sqlite3.connect(MONITOR_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT error_type, error_message, COUNT(*) as count
            FROM error_log
            WHERE DATE(timestamp) = ?
            GROUP BY error_type, error_message
            ORDER BY count DESC
            LIMIT 10
        ''', (date,))
        
        errors = cursor.fetchall()
        conn.close()
        
        return errors
    
    def get_top_users(self, date=None, limit=10):
        """Get most active users"""
        if date is None:
            date = (datetime.now() - timedelta(days=1)).date().isoformat()
        
        conn = sqlite3.connect(MONITOR_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, COUNT(*) as activity_count
            FROM activity_log
            WHERE DATE(timestamp) = ?
            GROUP BY user_id
            ORDER BY activity_count DESC
            LIMIT ?
        ''', (date, limit))
        
        users = cursor.fetchall()
        conn.close()
        
        return users
    
    async def generate_daily_report(self):
        """Generate comprehensive daily report"""
        yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
        stats = self.get_daily_stats(yesterday)
        errors = self.get_error_summary(yesterday)
        top_users = self.get_top_users(yesterday)
        
        # Build report
        report = f"""
📊 **DAILY BOT ACTIVITY REPORT**
📅 Date: {yesterday}
{'='*50}

👥 **USER STATISTICS**
━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Users: {stats['total_users']}
• New Users: {stats['new_users']}
• Total Requests: {stats['total_requests']}
• Success Rate: {(stats['successful_requests']/stats['total_requests']*100) if stats['total_requests'] > 0 else 0:.1f}%

📊 **ACTIVITY BREAKDOWN**
━━━━━━━━━━━━━━━━━━━━━━━━━━
• ✅ Successful: {stats['successful_requests']}
• ❌ Failed: {stats['failed_requests']}
• ⬆️ Uploads: {stats['total_uploads']}
• ⬇️ Downloads: {stats['total_downloads']}
• 💾 Data Processed: {stats['total_data_mb']:.2f} MB
• ⚡ Avg Processing Time: {stats['avg_processing_time']:.2f}s

🔥 **MOST USED FEATURES**
━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, (feature, count) in enumerate(stats['feature_stats'][:5], 1):
            report += f"{i}. {feature}: {count} times\n"
        
        report += f"""
👑 **TOP ACTIVE USERS**
━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, (user_id, username, count) in enumerate(top_users[:5], 1):
            username_display = username if username else f"User {user_id}"
            report += f"{i}. @{username_display}: {count} actions\n"
        
        if stats['errors_count'] > 0:
            report += f"""
⚠️ **ERROR SUMMARY** ({stats['errors_count']} total)
━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            for error_type, error_msg, count in errors[:5]:
                report += f"• {error_type}: {error_msg[:50]}... ({count}x)\n"
        else:
            report += "\n✅ **NO ERRORS** - Perfect day!\n"
        
        report += f"""
{'='*50}
📈 **TREND ANALYSIS**
"""
        
        # Get week comparison
        week_ago = (datetime.now() - timedelta(days=8)).date().isoformat()
        week_stats = self.get_daily_stats(week_ago)
        
        if week_stats['total_users'] > 0:
            user_growth = ((stats['total_users'] - week_stats['total_users']) / week_stats['total_users'] * 100)
            report += f"• User Growth (vs last week): {user_growth:+.1f}%\n"
        
        if week_stats['total_requests'] > 0:
            request_growth = ((stats['total_requests'] - week_stats['total_requests']) / week_stats['total_requests'] * 100)
            report += f"• Request Growth: {request_growth:+.1f}%\n"
        
        report += f"\n🤖 Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return report
    
    async def send_daily_report_to_admin(self):
        """Send daily report to admin via Telegram"""
        if not ADMIN_ID or not BOT_TOKEN:
            print("❌ Admin ID or Bot Token not configured")
            return
        
        try:
            bot = Bot(token=BOT_TOKEN)
            report = await self.generate_daily_report()
            
            # Split report if too long (Telegram limit is 4096 chars)
            max_length = 4000
            if len(report) > max_length:
                parts = [report[i:i+max_length] for i in range(0, len(report), max_length)]
                for part in parts:
                    await bot.send_message(chat_id=ADMIN_ID, text=part, parse_mode='Markdown')
                    await asyncio.sleep(1)
            else:
                await bot.send_message(chat_id=ADMIN_ID, text=report, parse_mode='Markdown')
            
            print(f"✅ Daily report sent to admin {ADMIN_ID}")
            
        except TelegramError as e:
            print(f"❌ Failed to send report: {e}")
        except Exception as e:
            print(f"❌ Error generating report: {e}")

# Standalone script to run daily
async def main():
    monitor = BotMonitor()
    await monitor.send_daily_report_to_admin()

if __name__ == '__main__':
    asyncio.run(main())
