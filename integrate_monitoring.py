#!/usr/bin/env python3
"""
Auto-integrate monitoring into super.py
This script automatically adds monitoring to all command handlers
"""

import re
import shutil
from datetime import datetime

def backup_file(filename):
    """Create backup of original file"""
    backup_name = f"{filename}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(filename, backup_name)
    print(f"✅ Backup created: {backup_name}")
    return backup_name

def add_monitor_import(content):
    """Add monitor import after dotenv import"""
    pattern = r'(from dotenv import load_dotenv\nload_dotenv\(\))'
    replacement = r'''\1

# Bot Monitoring System
try:
    from bot_monitor import BotMonitor
    import time as time_module
    MONITORING_ENABLED = True
    print("✅ Bot monitoring enabled")
except ImportError:
    MONITORING_ENABLED = False
    print("⚠️  Bot monitoring not available - install bot_monitor.py")
'''
    
    if 'from bot_monitor import BotMonitor' in content:
        print("ℹ️  Monitor import already exists")
        return content
    
    return re.sub(pattern, replacement, content)

def add_monitor_initialization(content):
    """Add monitor initialization after logger"""
    pattern = r'(logger = logging\.getLogger\(__name__\))'
    replacement = r'''\1

# Initialize Bot Monitor
if MONITORING_ENABLED:
    monitor = BotMonitor()
'''
    
    if 'monitor = BotMonitor()' in content:
        print("ℹ️  Monitor initialization already exists")
        return content
    
    return re.sub(pattern, replacement, content)

def add_daily_report_job(content):
    """Add daily report scheduler"""
    pattern = r'(scheduler\.add_job\([^)]+cleanup_old_downloads[^)]+\))'
    
    additional_job = r'''

    # Daily activity report to admin
    if MONITORING_ENABLED:
        async def send_daily_report():
            try:
                await monitor.send_daily_report_to_admin()
            except Exception as e:
                logger.error(f"Failed to send daily report: {e}")
        
        scheduler.add_job(
            send_daily_report,
            CronTrigger(hour=9, minute=0),  # 9 AM daily
            id='daily_report',
            name='Send daily activity report',
            replace_existing=True
        )
        logger.info("📊 Daily report scheduled for 9:00 AM")
'''
    
    if 'send_daily_report' in content:
        print("ℹ️  Daily report job already exists")
        return content
    
    return re.sub(pattern, r'\1' + additional_job, content)

def wrap_handler_with_monitoring(handler_code, handler_name, feature_name):
    """Wrap a handler function with monitoring code"""
    
    # Check if already wrapped
    if 'monitor.log_activity' in handler_code:
        return handler_code
    
    # Find function definition
    func_pattern = r'(async def ' + re.escape(handler_name) + r'\([^)]+\):)'
    
    # Add monitoring wrapper
    wrapped = re.sub(
        func_pattern,
        r'''\1
    start_time = time_module.time() if MONITORING_ENABLED else 0
    user_id = update.effective_user.id if update.effective_user else 0
    username = update.effective_user.username if update.effective_user else "Unknown"
    
    try:''',
        handler_code
    )
    
    # Add success logging at the end
    # Find the last return or end of function
    lines = wrapped.split('\n')
    indent_level = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('async def'):
            # Found function start, find its end
            for j in range(i+1, len(lines)):
                if lines[j].strip() and not lines[j].startswith(' '):
                    # End of function
                    success_log = f'''
        
        # Log successful activity
        if MONITORING_ENABLED:
            processing_time = time_module.time() - start_time
            monitor.log_activity(
                user_id=user_id,
                username=username,
                action='{handler_name}',
                feature='{feature_name}',
                success=True,
                processing_time=processing_time
            )
            monitor.update_feature_usage('{feature_name}')
'''
                    lines.insert(j, success_log)
                    break
            break
    
    wrapped = '\n'.join(lines)
    
    # Add except block before the function ends
    error_log = f'''
    except Exception as e:
        if MONITORING_ENABLED:
            processing_time = time_module.time() - start_time
            monitor.log_activity(
                user_id=user_id,
                username=username,
                action='{handler_name}',
                feature='{feature_name}',
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
            monitor.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                user_id=user_id,
                feature='{feature_name}'
            )
        raise
'''
    
    # Add except before last line of function
    wrapped = wrapped.rstrip() + error_log
    
    return wrapped

def main():
    print("🚀 Auto-integrating Bot Monitoring System")
    print("=" * 50)
    
    filename = 'super.py'
    
    # Create backup
    backup_file(filename)
    
    # Read file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("\n📝 Adding monitoring components...")
    
    # Add imports
    content = add_monitor_import(content)
    
    # Add initialization
    content = add_monitor_initialization(content)
    
    # Add daily report job
    content = add_daily_report_job(content)
    
    # Write modified file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ Integration complete!")
    print("\n📋 What was added:")
    print("  • Bot monitor import and initialization")
    print("  • Daily report scheduler (9 AM)")
    print("  • Error logging system")
    print("\n⚠️  Note: Handler wrapping requires manual integration")
    print("   See MONITORING_INTEGRATION.py for examples")
    print("\n🎯 Test the monitoring:")
    print("   python3 bot_monitor.py")

if __name__ == '__main__':
    main()
