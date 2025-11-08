# super_bot_interactive_admin.py

# Fix timezone issue BEFORE ANY OTHER IMPORTS
import os
import sys

# Set timezone to UTC using multiple methods
os.environ['TZ'] = 'UTC'
os.environ['TIMEZONE'] = 'UTC'

# On Windows, also try:
try:
    import time
    time.tzset() if hasattr(time, 'tzset') else None
except:
    pass

# --- 1. Imports ---
import logging
import io
import sqlite3
import secrets
import string
import asyncio
from datetime import datetime
from functools import wraps
from collections import defaultdict, deque
from typing import Dict, Optional, Set
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import functools
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Bot Monitoring System (Optional)
try:
    from bot_monitor import BotMonitor
    import time as time_module
    MONITORING_ENABLED = True
except ImportError:
    MONITORING_ENABLED = False
    class BotMonitor:
        """Dummy class when monitoring is not installed"""
        def log_activity(self, *args, **kwargs): pass
        def log_error(self, *args, **kwargs): pass
        def update_feature_usage(self, *args, **kwargs): pass

# Fix timezone issue for APScheduler BEFORE importing telegram  
import pytz

# Set default timezone
UTC = pytz.UTC

# Comprehensive pre-import patching
import apscheduler.util
_orig_get_localzone = None

try:
    from tzlocal import get_localzone as _orig_get_localzone
except:
    pass

def always_utc():
    """Always return UTC timezone"""
    return pytz.UTC

def always_utc_astimezone(obj):
    """Always return UTC for astimezone"""
    if obj is None:
        return pytz.UTC
    if isinstance(obj, str):
        if obj.upper() == 'UTC':
            return pytz.UTC
        try:
            return pytz.timezone(obj)
        except:
            return pytz.UTC
    if hasattr(obj, 'zone'):
        return obj
    return pytz.UTC

# Patch everything before imports
apscheduler.util.get_localzone = always_utc
apscheduler.util.astimezone = always_utc_astimezone

# Mock tzlocal module to always return UTC
class MockTzLocal:
    @staticmethod
    def get_localzone():
        return pytz.UTC
    
    @staticmethod
    def get_localzone_name():
        return 'UTC'

sys.modules['tzlocal'] = MockTzLocal()

# Import and comprehensively patch apscheduler BEFORE telegram import
try:
    import tzlocal
    tzlocal.get_localzone = lambda: pytz.UTC
    tzlocal.get_localzone_name = lambda: 'UTC'
except:
    pass

try:
    # Patch apscheduler.util module AGAIN (more comprehensive)
    import apscheduler.schedulers.base
    
    # Double-check patches are in place
    apscheduler.util.get_localzone = always_utc
    apscheduler.util.astimezone = always_utc_astimezone
    apscheduler.schedulers.base.get_localzone = always_utc
    
    # Also patch in the module's globals
    if hasattr(apscheduler.schedulers.base, '__dict__'):
        apscheduler.schedulers.base.__dict__['get_localzone'] = always_utc
        apscheduler.schedulers.base.__dict__['astimezone'] = always_utc_astimezone
    
    # Patch the util module globals too
    if hasattr(apscheduler.util, '__dict__'):
        apscheduler.util.__dict__['get_localzone'] = always_utc
        apscheduler.util.__dict__['astimezone'] = always_utc_astimezone
    
    print("✓ APScheduler timezone patches applied successfully")
    
except Exception as e:
    print(f"⚠ Warning: Could not patch apscheduler: {e}")

# Import all installed libraries
try:
    import httpx
    import pytz  # For timezone support
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
    from telegram.constants import ParseMode
    from telegram.error import Forbidden

    from moviepy.video.io.VideoFileClip import VideoFileClip
    from PIL import Image
    import fitz
    from docx import Document
    from gtts import gTTS
    # from googletrans import Translator  # Replaced with deep-translator
    from deep_translator import GoogleTranslator
    import yt_dlp
    import pyshorteners
    import qrcode
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from cryptography.hazmat.primitives import hashes
    import base64
except ImportError:
    print("Some libraries are not installed. Please run the pip install command from the previous step.")
    exit()

# --- 2. Configuration ---
# Load from .env file
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Parse ADMIN_ID safely
try:
    admin_id_str = os.getenv("ADMIN_USER_ID", "0")
    ADMIN_ID = int(admin_id_str) if admin_id_str.isdigit() else 0
except (ValueError, AttributeError):
    ADMIN_ID = 0

# Validate configuration
if not BOT_TOKEN or BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
    print("\n" + "="*60)
    print("❌ ERROR: Bot token not configured!")
    print("="*60)
    print("\n📋 Setup Instructions:")
    print("1. Copy .env.example to .env:")
    print("   cp .env.example .env")
    print("\n2. Edit .env file:")
    print("   nano .env")
    print("\n3. Set your credentials:")
    print("   BOT_TOKEN=your_bot_token_from_@BotFather")
    print("   ADMIN_USER_ID=your_telegram_user_id")
    print("\n4. Save and run again")
    print("="*60 + "\n")
    exit(1)

if ADMIN_ID == 0:
    print("\n" + "="*60)
    print("⚠️  WARNING: ADMIN_USER_ID not set!")
    print("="*60)
    print("Admin features will be disabled.")
    print("Get your user ID from @userinfobot and add to .env")
    print("="*60 + "\n")

# FFmpeg path configuration - NO ADMIN RIGHTS NEEDED!
# Try WinGet installation first, then system PATH
FFMPEG_PATH = os.path.join(os.environ.get('LOCALAPPDATA', ''), 
                          'Microsoft', 'WinGet', 'Packages', 
                          'Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe',
                          'ffmpeg-8.0-full_build', 'bin', 'ffmpeg.exe')

# Add ffmpeg to PATH for this process if it exists (no admin needed - user-level only)
if os.path.exists(FFMPEG_PATH):
    ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
    if ffmpeg_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
    # Set IMAGEIO_FFMPEG_EXE for moviepy
    os.environ['IMAGEIO_FFMPEG_EXE'] = FFMPEG_PATH
else:
    # Fallback: Try to use system FFmpeg (if available in PATH)
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=2)
        if result.returncode == 0:
            print("✓ Using system FFmpeg from PATH")
    except:
        print("⚠ FFmpeg not found - video/audio features may not work")
        print("  Install FFmpeg: apt-get install ffmpeg (Linux) or download from ffmpeg.org")

# --- 2.5. Google Drive Setup ---
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Google Drive API configuration
SCOPES = ['https://www.googleapis.com/auth/drive.file']
DRIVE_FOLDER_ID = None  # Optional: Set a specific folder ID to upload to
CREDENTIALS_FILE = 'credentials.json'  # Service account or OAuth credentials
TOKEN_FILE = 'token.json'  # For OAuth2 token storage

# --- 3. Basic Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Bot Monitor
if MONITORING_ENABLED:
    try:
        monitor = BotMonitor()
        logger.info("✅ Bot monitoring enabled")
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize monitoring: {e}")
        MONITORING_ENABLED = False
        monitor = BotMonitor()  # Use dummy
else:
    monitor = BotMonitor()  # Use dummy class
    logger.info("⚠️  Bot monitoring disabled (bot_monitor.py not found)")

# Thread pool for CPU-intensive operations (file processing, downloads, etc.)
# This allows true parallel processing for multiple users
executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix='BotWorker')

def run_in_executor(func):
    """Decorator to run blocking functions in thread pool"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))
    return wrapper

# --- 3.5. Optimized Data Structures for Multi-User Handling ---

class UserSessionManager:
    """Thread-safe session manager with O(1) access time"""
    def __init__(self):
        self._sessions: Dict[int, Dict] = {}
        self._lock = threading.RLock()
        
    def get_session(self, user_id: int) -> Dict:
        with self._lock:
            if user_id not in self._sessions:
                self._sessions[user_id] = {
                    'state': None,
                    'data': {},
                    'last_activity': datetime.now()
                }
            else:
                self._sessions[user_id]['last_activity'] = datetime.now()
            return self._sessions[user_id]
    
    def set_state(self, user_id: int, state: str, data: Optional[Dict] = None):
        with self._lock:
            session = self.get_session(user_id)
            session['state'] = state
            if data:
                session['data'].update(data)
    
    def clear_state(self, user_id: int):
        with self._lock:
            if user_id in self._sessions:
                self._sessions[user_id]['state'] = None
                self._sessions[user_id]['data'].clear()
    
    def get_state(self, user_id: int) -> Optional[str]:
        with self._lock:
            return self._sessions.get(user_id, {}).get('state')


class RateLimiter:
    """Token bucket algorithm for rate limiting - O(1) operations"""
    def __init__(self, rate: int = 20, per: int = 60):
        self._buckets: Dict[int, Dict] = defaultdict(lambda: {
            'tokens': rate,
            'last_update': time.time(),
            'rate': rate,
            'per': per
        })
        self._lock = threading.RLock()
    
    def consume(self, user_id: int, tokens: int = 1) -> bool:
        with self._lock:
            bucket = self._buckets[user_id]
            now = time.time()
            elapsed = now - bucket['last_update']
            
            # Refill tokens
            tokens_to_add = (elapsed / bucket['per']) * bucket['rate']
            bucket['tokens'] = min(bucket['rate'], bucket['tokens'] + tokens_to_add)
            bucket['last_update'] = now
            
            # Try to consume
            if bucket['tokens'] >= tokens:
                bucket['tokens'] -= tokens
                return True
            return False
    
    def get_wait_time(self, user_id: int) -> float:
        with self._lock:
            bucket = self._buckets[user_id]
            if bucket['tokens'] >= 1:
                return 0.0
            tokens_needed = 1 - bucket['tokens']
            return (tokens_needed / bucket['rate']) * bucket['per']


class TaskLockManager:
    """Manages user task locks to prevent concurrent operations"""
    def __init__(self):
        self._active_tasks: Dict[int, str] = {}  # user_id -> task_name
        self._lock = threading.RLock()
    
    def is_user_busy(self, user_id: int) -> bool:
        """Check if user has an active task"""
        with self._lock:
            return user_id in self._active_tasks
    
    def get_active_task(self, user_id: int) -> Optional[str]:
        """Get the name of user's active task"""
        with self._lock:
            return self._active_tasks.get(user_id)
    
    def lock_user(self, user_id: int, task_name: str) -> bool:
        """Lock user for a task. Returns True if locked, False if already busy"""
        with self._lock:
            if user_id in self._active_tasks:
                return False
            self._active_tasks[user_id] = task_name
            return True
    
    def unlock_user(self, user_id: int):
        """Unlock user after task completion"""
        with self._lock:
            if user_id in self._active_tasks:
                del self._active_tasks[user_id]
    
    def get_busy_users_count(self) -> int:
        """Get count of users with active tasks"""
        with self._lock:
            return len(self._active_tasks)


class DownloadQueue:
    """Priority queue for download management - O(log n) insertion"""
    def __init__(self, max_concurrent: int = 3):
        self._queue: deque = deque()
        self._active: Set[str] = set()
        self._lock = asyncio.Lock()
        self.max_concurrent = max_concurrent
        self._counter = 0
    
    async def add_download(self, user_id: int, url: str, priority: int = 5):
        async with self._lock:
            self._counter += 1
            download_id = f"{user_id}_{self._counter}"
            self._queue.append({
                'id': download_id,
                'user_id': user_id,
                'url': url,
                'priority': priority,
                'added_at': datetime.now()
            })
            return download_id
    
    async def can_download(self) -> bool:
        async with self._lock:
            return len(self._active) < self.max_concurrent
    
    async def start_download(self, download_id: str):
        async with self._lock:
            self._active.add(download_id)
    
    async def complete_download(self, download_id: str):
        async with self._lock:
            self._active.discard(download_id)
    
    def get_stats(self) -> Dict:
        return {
            'queue_size': len(self._queue),
            'active': len(self._active),
            'available_slots': self.max_concurrent - len(self._active)
        }


# Initialize global optimized data structures
session_manager = UserSessionManager()
rate_limiter = RateLimiter(rate=20, per=60)  # 20 actions per minute per user
download_queue = DownloadQueue(max_concurrent=5)  # 5 concurrent downloads
task_lock = TaskLockManager()  # Manage user task locks

# --- URL Storage for Download Callbacks ---
# Store URLs temporarily to avoid callback_data length limits (64 bytes max)
user_download_urls = {}  # {user_id: url}

def store_download_url(user_id: int, url: str) -> None:
    """Store URL for user temporarily"""
    user_download_urls[user_id] = url

def get_download_url(user_id: int) -> Optional[str]:
    """Retrieve and remove stored URL"""
    return user_download_urls.pop(user_id, None)

def escape_markdown(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2"""
    if not text:
        return ""
    # Escape special Markdown characters
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text

def compress_video(input_file: str, target_size_mb: int = 50) -> str:
    """Compress video to target size using FFmpeg with FAST optimized settings"""
    try:
        import subprocess
        
        output_file = input_file.rsplit('.', 1)[0] + '_compressed.mp4'
        
        # Get video duration
        probe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 
            'format=duration', '-of', 
            'default=noprint_wrappers=1:nokey=1', input_file
        ]
        duration = float(subprocess.check_output(probe_cmd).decode().strip())
        
        # Calculate target bitrate (70% of target for safety margin)
        target_size_bits = target_size_mb * 8 * 1024 * 1024 * 0.70
        target_bitrate = int(target_size_bits / duration)
        
        # Lower audio bitrate for files that need heavy compression
        audio_bitrate = '96k' if target_size_mb < 35 else '128k'
        
        # Compress with FFmpeg - FAST settings optimized for speed
        compress_cmd = [
            'ffmpeg', '-i', input_file, 
            '-c:v', 'libx264',
            '-b:v', f'{target_bitrate}',
            '-c:a', 'aac',
            '-b:a', audio_bitrate,
            '-preset', 'veryfast',  # MUCH faster than 'medium' or 'fast'
            '-crf', '28',  # Quality control
            '-threads', '0',  # Use all CPU cores
            '-y', output_file
        ]
        subprocess.run(compress_cmd, check=True, capture_output=True)
        
        return output_file
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        return input_file

def split_file(input_file: str, chunk_size_mb: int = 50) -> list:
    """Split large file into smaller chunks"""
    try:
        chunk_size = chunk_size_mb * 1024 * 1024
        chunks = []
        
        with open(input_file, 'rb') as f:
            chunk_num = 0
            while True:
                chunk_data = f.read(chunk_size)
                if not chunk_data:
                    break
                
                chunk_file = f"{input_file}.part{chunk_num}"
                with open(chunk_file, 'wb') as chunk_f:
                    chunk_f.write(chunk_data)
                
                chunks.append(chunk_file)
                chunk_num += 1
        
        return chunks
    except Exception as e:
        logger.error(f"File splitting failed: {e}")
        return [input_file]

# --- Google Drive Upload Functions ---
def get_drive_service():
    """Authenticate and return Google Drive service"""
    creds = None
    
    # Check if token file exists (OAuth2) - Try this FIRST
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            logger.info("Using existing OAuth2 token")
        except Exception as e:
            logger.warning(f"Failed to load token: {e}")
            creds = None
    
    # Refresh expired credentials
    if creds and creds.expired and creds.refresh_token:
        try:
            logger.info("Token expired, refreshing...")
            creds.refresh(Request())
            logger.info("✓ OAuth2 token refreshed successfully")
            # Save refreshed token
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        except Exception as e:
            logger.error(f"❌ Token refresh failed: {e}")
            logger.error("Please generate a new token.json file")
            creds = None
    
    # If no valid OAuth2 credentials, try to create them
    if not creds or not creds.valid:
        if os.path.exists(CREDENTIALS_FILE):
            try:
                # Check if it's a service account file
                with open(CREDENTIALS_FILE, 'r') as f:
                    import json
                    cred_data = json.load(f)
                    
                if cred_data.get('type') == 'service_account':
                    logger.error("Service accounts don't have storage quota!")
                    logger.error("Please create OAuth2 credentials instead.")
                    logger.error("See GOOGLE_DRIVE_SETUP.md for OAuth2 setup instructions.")
                    return None
                else:
                    # It's OAuth2 credentials - run authorization flow
                    logger.error("="*60)
                    logger.error("❌ OAuth2 authorization required!")
                    logger.error("="*60)
                    logger.error("token.json is missing or invalid.")
                    logger.error("")
                    logger.error("On a headless server (no browser), you need to:")
                    logger.error("1. Generate token.json on your local PC")
                    logger.error("2. Copy it to the server")
                    logger.error("")
                    logger.error("Steps:")
                    logger.error("  - Run bot on Windows/Mac (with browser)")
                    logger.error("  - Authorize Google Drive access")
                    logger.error("  - Copy the generated token.json to server")
                    logger.error("="*60)
                    return None
            except Exception as e:
                logger.error(f"Failed to authenticate: {e}")
                return None
        else:
            logger.error(f"No credentials file found. Please create {CREDENTIALS_FILE}")
            return None
    
    return build('drive', 'v3', credentials=creds)

def upload_to_google_drive(file_path: str, file_name: str, progress_callback=None) -> str:
    """
    Upload file to Google Drive and return shareable link
    
    Args:
        file_path: Local path to the file
        file_name: Name to use in Google Drive
        progress_callback: Optional callback function(current_bytes, total_bytes)
    
    Returns:
        Shareable download link or None if failed
    """
    try:
        service = get_drive_service()
        if not service:
            logger.error("Failed to authenticate with Google Drive")
            return None
        
        # File metadata
        file_metadata = {
            'name': file_name,
        }
        
        # If folder ID is specified, upload to that folder
        if DRIVE_FOLDER_ID:
            file_metadata['parents'] = [DRIVE_FOLDER_ID]
        
        # Detect MIME type
        mime_type = 'video/mp4'  # Default for videos
        if file_name.lower().endswith('.mp3'):
            mime_type = 'audio/mpeg'
        elif file_name.lower().endswith('.webm'):
            mime_type = 'video/webm'
        elif file_name.lower().endswith('.mkv'):
            mime_type = 'video/x-matroska'
        
        # Dynamic chunk size for faster uploads
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:  # Files > 100MB
            chunk_size = 50 * 1024 * 1024  # 50 MB chunks
        elif file_size > 50 * 1024 * 1024:  # Files > 50MB
            chunk_size = 25 * 1024 * 1024  # 25 MB chunks
        else:
            chunk_size = 10 * 1024 * 1024  # 10 MB chunks for smaller files
        
        # Upload file with resumable upload for large files
        media = MediaFileUpload(
            file_path,
            mimetype=mime_type,
            resumable=True,
            chunksize=chunk_size
        )
        
        logger.info(f"Uploading {file_name} ({file_size/1024/1024:.1f} MB) to Google Drive with {chunk_size/1024/1024:.0f} MB chunks...")
        
        # Create the file with resumable upload
        request = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink, webContentLink'
        )
        
        # Upload with progress tracking
        response = None
        uploaded_size = 0
        
        while response is None:
            status, response = request.next_chunk()
            if status:
                uploaded_size = int(status.resumable_progress)
                if progress_callback:
                    progress_callback(uploaded_size, file_size)
                logger.debug(f"Upload progress: {int(status.progress() * 100)}%")
        
        file = response
        file_id = file.get('id')
        
        # Make file publicly accessible
        service.permissions().create(
            fileId=file_id,
            body={'type': 'anyone', 'role': 'reader'}
        ).execute()
        
        # Generate multiple download link formats
        # Method 1: Direct download (works best for smaller files <100MB)
        direct_download = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
        
        # Method 2: Force download with alternative URL (bypasses processing page)
        force_download = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        
        # Method 3: View/Stream link (requires processing)
        view_link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        
        logger.info(f"Upload successful! File ID: {file_id}")
        logger.info(f"Force download link: {force_download}")
        logger.info(f"Direct download link: {direct_download}")
        logger.info(f"View link: {view_link}")
        
        # Return the force download link (best for large files)
        return force_download
        
    except Exception as e:
        logger.error(f"Google Drive upload failed: {e}")
        return None

async def drivetest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick connectivity and permission test for Google Drive."""
    await update.message.reply_text("🔍 Running Google Drive connectivity test... Please wait...")
    service = get_drive_service()
    if not service:
        await update.message.reply_text(
            "❌ Drive auth failed. Ensure credentials.json exists and run an OAuth authorization.")
        return
    try:
        # List first 5 files
        results = service.files().list(pageSize=5, fields="files(id,name,mimeType,createdTime)").execute()
        files = results.get('files', [])
        listing = "\n".join([f"• {f['name']} ({f['id']})" for f in files]) or "(No files visible)"
        # Create small temp file
        temp_name = f"drive_test_{int(time.time())}.txt"
        with open(temp_name, 'w', encoding='utf-8') as f:
            f.write("Drive API test at " + datetime.now().isoformat())
        media = MediaFileUpload(temp_name, resumable=False)
        meta = {'name': temp_name}
        created = service.files().create(body=meta, media_body=media, fields='id').execute()
        file_id = created.get('id')
        # Make public
        service.permissions().create(fileId=file_id, body={'type': 'anyone', 'role': 'reader'}).execute()
        view_link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        # Clean up
        service.files().delete(fileId=file_id).execute()
        os.remove(temp_name)
        await update.message.reply_text(
            "✅ Drive test passed!\n\n" +
            "📁 Sample file created & deleted successfully.\n" +
            "🔗 Public link (during test):\n" + view_link + "\n\n" +
            "📋 Listing (first 5 files):\n" + listing
        )
    except Exception as e:
        import traceback
        await update.message.reply_text("❌ Drive test failed: " + str(e))
        logger.error("Drive test error:\n" + traceback.format_exc())

# --- ShrinkMe.io Link Shortener ---
def shorten_link_shrinkme(long_url: str, custom_alias: str = "") -> str:
    """
    Shorten URL using ShrinkMe.io API
    
    Args:
        long_url: The long URL to shorten
        custom_alias: Optional custom alias for the short link
    
    Returns:
        Shortened URL or original URL if shortening fails
    """
    try:
        import requests
        
        # ShrinkMe.io API configuration
        api_key = "325720a06a3a8a587755631e41dbb13e64822e0f"
        api_url = "https://shrinkme.io/api"
        
        # Build API request URL
        params = {
            'api': api_key,
            'url': long_url
        }
        
        # Add custom alias if provided
        if custom_alias:
            params['alias'] = custom_alias
        
        # Make API request
        response = requests.get(api_url, params=params, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if shortening was successful
            if result.get('status') == 'success':
                short_url = result.get('shortenedUrl')
                logger.info(f"Link shortened: {long_url[:50]}... → {short_url}")
                return short_url
            else:
                error_msg = result.get('message', 'Unknown error')
                logger.warning(f"ShrinkMe.io API error: {error_msg}")
                return long_url
        else:
            logger.warning(f"ShrinkMe.io API returned status {response.status_code}")
            return long_url
            
    except Exception as e:
        logger.error(f"Link shortening failed: {e}")
        return long_url  # Return original URL if shortening fails

# --- Automatic File Cleanup System ---
async def cleanup_old_downloads():
    """Periodically clean up old download files"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            
            current_time = time.time()
            cleaned_count = 0
            
            # Find and delete files older than 10 minutes
            for filename in os.listdir('.'):
                if filename.endswith(('.mp4', '.mp3', '.webm', '.m4a', '.mkv', '.avi')):
                    try:
                        file_age = current_time - os.path.getmtime(filename)
                        if file_age > 600:  # 10 minutes
                            os.remove(filename)
                            cleaned_count += 1
                            logger.info(f"🗑️ Auto-cleaned old file: {filename}")
                    except Exception as e:
                        logger.debug(f"Could not clean {filename}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"✅ Auto-cleanup: Removed {cleaned_count} old file(s)")
                
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")

# --- 4. Database Setup ---
class DatabasePool:
    """Connection pool for efficient database operations"""
    def __init__(self, db_path: str, pool_size: int = 3):
        self.db_path = db_path
        self._pool = deque()
        self._lock = threading.Lock()
        
        # Create connection pool
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._pool.append(conn)
    
    def get_connection(self):
        with self._lock:
            if self._pool:
                return self._pool.popleft()
            # Create new connection if pool is empty
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
    
    def return_connection(self, conn):
        with self._lock:
            self._pool.append(conn)
    
    def execute(self, query: str, params: tuple = (), fetch: str = 'all'):
        """Execute query with automatic connection management"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if fetch == 'all':
                result = cursor.fetchall()
            elif fetch == 'one':
                result = cursor.fetchone()
            else:
                result = cursor.lastrowid
            
            conn.commit()
            return result
        finally:
            self.return_connection(conn)

# Initialize database pool with increased size for concurrent users
db_pool = DatabasePool('bot_database.db', pool_size=10)  # Increased from 3 to 10

def init_db():
    """Initializes the SQLite database and creates tables if they don't exist."""
    with sqlite3.connect('bot_database.db') as conn:
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrent access
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('PRAGMA synchronous=NORMAL')
        cursor.execute('PRAGMA cache_size=10000')
        cursor.execute('PRAGMA temp_store=MEMORY')
        
        # Existing tables
        cursor.execute('CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, note TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
        cursor.execute('CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY, first_name TEXT, username TEXT, start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP, verified INTEGER DEFAULT 0)')
        
        # Add verified column to existing users table if it doesn't exist
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN verified INTEGER DEFAULT 0')
        except sqlite3.OperationalError:
            # Column already exists, ignore the error
            pass
        
        # NEW: User usage tracking table
        cursor.execute('''CREATE TABLE IF NOT EXISTS user_usage (
            user_id INTEGER PRIMARY KEY,
            usage_count INTEGER DEFAULT 0,
            last_reset_date DATE DEFAULT CURRENT_DATE,
            premium_status INTEGER DEFAULT 0,
            premium_until DATE DEFAULT NULL
        )''')
        
        # NEW: Pending payments table
        cursor.execute('''CREATE TABLE IF NOT EXISTS pending_payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            username TEXT,
            first_name TEXT,
            reference_number TEXT NOT NULL,
            screenshot_file_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending'
        )''')
        
        # NEW: Settings table for payment configuration
        cursor.execute('''CREATE TABLE IF NOT EXISTS bot_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )''')
        
        # NEW: Banned users table
        cursor.execute('''CREATE TABLE IF NOT EXISTS banned_users (
            user_id INTEGER PRIMARY KEY,
            banned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reason TEXT DEFAULT NULL
        )''')
        
        # NEW: Tool usage statistics table
        cursor.execute('''CREATE TABLE IF NOT EXISTS tool_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            tool_name TEXT NOT NULL,
            used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # NEW: File storage table
        cursor.execute('''CREATE TABLE IF NOT EXISTS stored_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            drive_file_id TEXT NOT NULL,
            share_link TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Initialize default settings
        cursor.execute("INSERT OR IGNORE INTO bot_settings (key, value) VALUES ('upi_id', 'yourpayments@paytm')")
        cursor.execute("INSERT OR IGNORE INTO bot_settings (key, value) VALUES ('qr_code_file_id', '')")
        cursor.execute("INSERT OR IGNORE INTO bot_settings (key, value) VALUES ('premium_price', '99')")
        cursor.execute("INSERT OR IGNORE INTO bot_settings (key, value) VALUES ('free_usage_limit', '5')")
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_notes_user_id ON notes(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_usage_user_id ON user_usage(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pending_payments_user_id ON pending_payments(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pending_payments_status ON pending_payments(status)')
        
        conn.commit()

# --- 4.5. Usage Tracking Functions ---
def check_and_reset_usage(user_id: int):
    """Check if usage needs to be reset (24 hours passed) and reset if needed"""
    from datetime import date, timedelta
    
    try:
        user_data = db_pool.execute(
            "SELECT usage_count, last_reset_date, premium_status FROM user_usage WHERE user_id = ?",
            (user_id,),
            fetch='one'
        )
        
        if not user_data:
            # Create new user usage entry
            db_pool.execute(
                "INSERT INTO user_usage (user_id, usage_count, last_reset_date, premium_status) VALUES (?, 0, CURRENT_DATE, 0)",
                (user_id,),
                fetch='none'
            )
            return True  # Can use
        
        last_reset = datetime.strptime(user_data[1], '%Y-%m-%d').date()
        today = date.today()
        
        # Reset if 24 hours passed
        if today > last_reset:
            db_pool.execute(
                "UPDATE user_usage SET usage_count = 0, last_reset_date = CURRENT_DATE WHERE user_id = ?",
                (user_id,),
                fetch='none'
            )
            return True
        
        return True
    except Exception as e:
        logger.error(f"Error in check_and_reset_usage: {e}")
        return True

def is_user_verified(user_id: int) -> bool:
    """Check if user has completed verification"""
    try:
        result = db_pool.execute(
            "SELECT verified FROM users WHERE user_id = ?",
            (user_id,),
            fetch='one'
        )
        return result and result[0] == 1
    except Exception as e:
        logger.error(f"Error checking verification status: {e}")
        return False

def verify_user(user_id: int) -> bool:
    """Mark user as verified"""
    try:
        db_pool.execute(
            "UPDATE users SET verified = 1 WHERE user_id = ?",
            (user_id,),
            fetch='none'
        )
        return True
    except Exception as e:
        logger.error(f"Error verifying user: {e}")
        return False

def can_use_tool(user_id: int) -> tuple[bool, str]:
    """Check if user can use a tool. Returns (can_use, message)"""
    try:
        # Check if user is banned
        banned = db_pool.execute(
            "SELECT user_id FROM banned_users WHERE user_id = ?",
            (user_id,),
            fetch='one'
        )
        
        if banned:
            return False, (
                "🚫 **Access Restricted**\n\n"
                "Your access to this bot has been restricted.\n\n"
                "Contact the administrator for more information."
            )
        
        check_and_reset_usage(user_id)
        
        user_data = db_pool.execute(
            "SELECT usage_count, premium_status FROM user_usage WHERE user_id = ?",
            (user_id,),
            fetch='one'
        )
        
        if not user_data:
            return True, ""
        
        usage_count, premium_status = user_data[0], user_data[1]
        
        # Premium users have unlimited access
        if premium_status == 1:
            return True, ""
        
        # Free users have configurable limit per day
        free_limit = int(get_setting('free_usage_limit', '5'))
        if usage_count >= free_limit:
            return False, (
                "⚠️ **Daily Free Limit Reached!**\n\n"
                f"You've used all {free_limit} free tools today.\n\n"
                "💎 **Upgrade to Premium** for unlimited access!\n"
                "Click the button below to get premium."
            )
        
        return True, ""
    except Exception as e:
        logger.error(f"Error in can_use_tool: {e}")
        return True, ""  # Allow on error

def log_tool_usage(user_id: int, tool_name: str):
    """Log tool usage for statistics"""
    try:
        db_pool.execute(
            "INSERT INTO tool_usage (user_id, tool_name) VALUES (?, ?)",
            (user_id, tool_name),
            fetch='none'
        )
    except Exception as e:
        logger.error(f"Error logging tool usage: {e}")

def increment_usage(user_id: int):
    """Increment user's daily usage count"""
    try:
        check_and_reset_usage(user_id)
        db_pool.execute(
            "UPDATE user_usage SET usage_count = usage_count + 1 WHERE user_id = ?",
            (user_id,),
            fetch='none'
        )
    except Exception as e:
        logger.error(f"Error incrementing usage: {e}")

def get_user_stats(user_id: int) -> dict:
    """Get user statistics"""
    from datetime import datetime, date, time, timedelta
    
    try:
        check_and_reset_usage(user_id)
        
        user_data = db_pool.execute(
            "SELECT usage_count, last_reset_date, premium_status FROM user_usage WHERE user_id = ?",
            (user_id,),
            fetch='one'
        )
        
        if not user_data:
            return {
                'usage_count': 0,
                'premium': False,
                'remaining': 5,
                'reset_time': '24:00'
            }
        
        usage_count, last_reset, premium_status = user_data[0], user_data[1], user_data[2]
        
        # Calculate time until reset
        last_reset_date = datetime.strptime(last_reset, '%Y-%m-%d').date()
        tomorrow = last_reset_date + timedelta(days=1)
        now = datetime.now()
        reset_datetime = datetime.combine(tomorrow, time(0, 0))
        time_diff = reset_datetime - now
        
        hours = int(time_diff.total_seconds() // 3600)
        minutes = int((time_diff.total_seconds() % 3600) // 60)
        
        return {
            'usage_count': usage_count,
            'premium': premium_status == 1,
            'remaining': max(0, 5 - usage_count),
            'reset_time': f"{hours:02d}:{minutes:02d}"
        }
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return {
            'usage_count': 0,
            'premium': False,
            'remaining': 5,
            'reset_time': '24:00'
        }

def get_setting(key: str, default: str = '') -> str:
    """Get a bot setting value"""
    try:
        result = db_pool.execute(
            "SELECT value FROM bot_settings WHERE key = ?",
            (key,),
            fetch='one'
        )
        return result[0] if result else default
    except:
        return default

def set_setting(key: str, value: str):
    """Set a bot setting value"""
    try:
        db_pool.execute(
            "INSERT OR REPLACE INTO bot_settings (key, value) VALUES (?, ?)",
            (key, value),
            fetch='none'
        )
    except Exception as e:
        logger.error(f"Error setting value: {e}")

# --- 5. Admin Panel Implementation (NEW & IMPROVED) ---

# State constant for ConversationHandler
AWAITING_BROADCAST_MESSAGE = 'awaiting_broadcast_message'

def admin_only(func):
    """Decorator to restrict access to admin-only functions."""
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        if update.effective_user.id != ADMIN_ID:
            await update.message.reply_text("⛔️ You are not authorized to use this command.")
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

@admin_only
async def admin_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Displays the main admin panel keyboard."""
    keyboard = [
        [InlineKeyboardButton("📊 User Statistics", callback_data='admin_stats')],
        [InlineKeyboardButton("📈 Tool Usage Stats", callback_data='admin_tool_stats')],
        [InlineKeyboardButton("🧾 View Pending Payments", callback_data='admin_payments')],
        [InlineKeyboardButton("💎 View Premium Users", callback_data='admin_premium_users')],
        [
            InlineKeyboardButton("✅ Grant Premium", callback_data='admin_grant_premium'),
            InlineKeyboardButton("❌ Revoke Premium", callback_data='admin_revoke_premium')
        ],
        [
            InlineKeyboardButton("🚫 Ban User", callback_data='admin_ban_user'),
            InlineKeyboardButton("✅ Unban User", callback_data='admin_unban_user')
        ],
        [InlineKeyboardButton("🎯 Set Free Limit", callback_data='admin_set_limit')],
        [InlineKeyboardButton("💰 Payment Settings", callback_data='admin_payment_settings')],
        [InlineKeyboardButton("📣 Start Broadcast", callback_data='admin_broadcast_start')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("👑 *Admin Panel*", reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)

@admin_only
async def set_upi_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to set UPI ID"""
    if not context.args:
        await update.message.reply_text(
            "❌ **Usage:** `/setupi <UPI ID>`\n\n"
            "💡 **Example:** `/setupi yourpayments@paytm`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    upi_id = " ".join(context.args)
    set_setting('upi_id', upi_id)
    
    await update.message.reply_text(
        f"✅ **UPI ID Updated!**\n\n"
        f"New UPI ID: `{upi_id}`",
        parse_mode=ParseMode.MARKDOWN
    )

@admin_only
async def set_price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to set premium price"""
    if not context.args:
        await update.message.reply_text(
            "❌ **Usage:** `/setprice <amount>`\n\n"
            "💡 **Example:** `/setprice 99`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    try:
        price = int(context.args[0])
        set_setting('premium_price', str(price))
        
        await update.message.reply_text(
            f"✅ **Premium Price Updated!**\n\n"
            f"New Price: ₹{price}",
            parse_mode=ParseMode.MARKDOWN
        )
    except ValueError:
        await update.message.reply_text("❌ Please provide a valid number")

@admin_only
async def set_qr_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to set QR code"""
    session_manager.set_state(ADMIN_ID, 'awaiting_qr_code')
    
    await update.message.reply_text(
        "📸 **Please send the QR code image now.**\n\n"
        "This will be shown to users when they want to buy premium.",
        parse_mode=ParseMode.MARKDOWN
    )

async def show_payment_for_review(query, context):
    """Show a payment for admin review"""
    payments = context.user_data.get('pending_payments', [])
    index = context.user_data.get('current_payment_index', 0)
    
    if index >= len(payments):
        await query.edit_message_text("✅ All payments reviewed!")
        return
    
    payment = payments[index]
    
    # Create review message
    review_text = (
        f"🧾 **Payment Review** ({index + 1}/{len(payments)})\n\n"
        f"👤 **User:** {payment['first_name']}\n"
        f"🆔 **User ID:** `{payment['user_id']}`\n"
        f"👉 **Username:** @{payment['username'] or 'None'}\n"
        f"🔢 **Reference:** `{payment['reference_number']}`\n"
        f"🕐 **Time:** {payment['timestamp']}\n\n"
        f"📸 **Screenshot attached below**"
    )
    
    # Create navigation buttons
    buttons = []
    nav_row = []
    if index > 0:
        nav_row.append(InlineKeyboardButton("⬅️ Previous", callback_data='payment_prev'))
    if index < len(payments) - 1:
        nav_row.append(InlineKeyboardButton("Next ➡️", callback_data='payment_next'))
    
    if nav_row:
        buttons.append(nav_row)
    
    buttons.append([
        InlineKeyboardButton("✅ Accept", callback_data=f"payment_accept_{payment['id']}"),
        InlineKeyboardButton("❌ Reject", callback_data=f"payment_reject_{payment['id']}")
    ])
    buttons.append([InlineKeyboardButton("⬅️ Back to Admin", callback_data='admin_back')])
    
    # Send screenshot and review text
    try:
        await query.message.reply_photo(
            photo=payment['screenshot_file_id'],
            caption=review_text,
            reply_markup=InlineKeyboardMarkup(buttons),
            parse_mode=ParseMode.MARKDOWN
        )
        await query.delete_message()
    except:
        await query.edit_message_text(
            text=review_text + "\n\n❌ Screenshot unavailable",
            reply_markup=InlineKeyboardMarkup(buttons),
            parse_mode=ParseMode.MARKDOWN
        )

async def accept_payment(query, context, payment_id: int):
    """Accept a payment and grant premium"""
    try:
        # Get payment details
        payment = db_pool.execute(
            "SELECT user_id, first_name FROM pending_payments WHERE id = ?",
            (payment_id,),
            fetch='one'
        )
        
        if not payment:
            await query.answer("Payment not found!", show_alert=True)
            return
        
        user_id, first_name = payment[0], payment[1]
        
        # Grant premium status
        db_pool.execute(
            "INSERT OR REPLACE INTO user_usage (user_id, usage_count, last_reset_date, premium_status) VALUES (?, 0, CURRENT_DATE, 1)",
            (user_id,),
            fetch='none'
        )
        
        # Delete pending payment
        db_pool.execute(
            "DELETE FROM pending_payments WHERE id = ?",
            (payment_id,),
            fetch='none'
        )
        
        # Notify user
        try:
            await context.bot.send_message(
                chat_id=user_id,
                text=(
                    "✅ **Payment Accepted!**\n\n"
                    "🎉 Welcome to Premium!\n\n"
                    "You now have **unlimited access** to all bot features.\n"
                    "Thank you for your support! 💎"
                ),
                parse_mode=ParseMode.MARKDOWN
            )
        except:
            pass
        
        await query.answer("✅ Payment accepted! User granted premium.", show_alert=True)
        
        # Remove from list and show next
        payments = context.user_data.get('pending_payments', [])
        payments = [p for p in payments if p['id'] != payment_id]
        context.user_data['pending_payments'] = payments
        
        if payments:
            await show_payment_for_review(query, context)
        else:
            # Delete the message with photo and send a new text message
            try:
                await query.message.delete()
            except:
                pass
            await query.message.reply_text("✅ All payments processed!")
            
    except Exception as e:
        logger.error(f"Error accepting payment: {e}")
        await query.answer(f"❌ Error: {str(e)}", show_alert=True)

async def reject_payment(query, context, payment_id: int):
    """Reject a payment"""
    try:
        # Get payment details
        payment = db_pool.execute(
            "SELECT user_id, first_name FROM pending_payments WHERE id = ?",
            (payment_id,),
            fetch='one'
        )
        
        if not payment:
            await query.answer("Payment not found!", show_alert=True)
            return
        
        user_id, first_name = payment[0], payment[1]
        
        # Delete pending payment
        db_pool.execute(
            "DELETE FROM pending_payments WHERE id = ?",
            (payment_id,),
            fetch='none'
        )
        
        # Notify user
        try:
            await context.bot.send_message(
                chat_id=user_id,
                text=(
                    "❌ **Payment Not Verified**\n\n"
                    "We couldn't verify your payment.\n\n"
                    "Possible reasons:\n"
                    "• Incorrect reference number\n"
                    "• Payment not received\n"
                    "• Screenshot unclear\n\n"
                    "Please try again or contact support."
                ),
                parse_mode=ParseMode.MARKDOWN
            )
        except:
            pass
        
        await query.answer("Payment rejected and user notified.", show_alert=True)
        
        # Remove from list and show next
        payments = context.user_data.get('pending_payments', [])
        payments = [p for p in payments if p['id'] != payment_id]
        context.user_data['pending_payments'] = payments
        
        if payments:
            await show_payment_for_review(query, context)
        else:
            # Delete the message with photo and send a new text message
            try:
                await query.message.delete()
            except:
                pass
            await query.message.reply_text("✅ All payments processed!")
            
    except Exception as e:
        logger.error(f"Error rejecting payment: {e}")
        await query.answer(f"❌ Error: {str(e)}", show_alert=True)

async def admin_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles all admin panel button clicks."""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == 'admin_stats':
        try:
            user_count = db_pool.execute("SELECT COUNT(*) FROM users", fetch='one')[0]
            active_sessions = session_manager.get_session(ADMIN_ID)  # This creates/updates admin session
            
            # Get download queue stats
            dl_stats = download_queue.get_stats()
            
            stats_text = (
                f"📊 **Bot Statistics**\n\n"
                f"👥 **Users:**\n"
                f"• Total Users: `{user_count}`\n"
                f"• Active Sessions: `{len(session_manager._sessions)}`\n\n"
                f"📥 **Downloads:**\n"
                f"• Queue Size: `{dl_stats['queue_size']}`\n"
                f"• Active Downloads: `{dl_stats['active']}`\n"
                f"• Available Slots: `{dl_stats['available_slots']}/{download_queue.max_concurrent}`\n\n"
                f"⚡ **Performance:**\n"
                f"• Rate Limiter Active\n"
                f"• Connection Pool: 3 connections\n"
                f"• Concurrent Downloads: 5 max\n\n"
                f"🚀 **Optimized for Multi-User Efficiency!**"
            )
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            stats_text = f"📊 **Stats Error:** {str(e)}"
        
        await query.edit_message_text(text=stats_text, parse_mode=ParseMode.MARKDOWN)
    
    elif data == 'admin_payments':
        # View pending payments
        try:
            payments = db_pool.execute(
                "SELECT id, user_id, username, first_name, reference_number, screenshot_file_id, timestamp FROM pending_payments WHERE status='pending' ORDER BY timestamp DESC LIMIT 10",
                fetch='all'
            )
            
            if not payments:
                await query.edit_message_text(
                    "🧾 **No Pending Payments**\n\nThere are no payment requests to review.",
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                # Show first payment
                context.user_data['pending_payments'] = [dict(p) for p in payments]
                context.user_data['current_payment_index'] = 0
                await show_payment_for_review(query, context)
        except Exception as e:
            logger.error(f"Error fetching payments: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)}")
    
    elif data.startswith('payment_accept_'):
        payment_id = int(data.split('_')[2])
        await accept_payment(query, context, payment_id)
    
    elif data.startswith('payment_reject_'):
        payment_id = int(data.split('_')[2])
        await reject_payment(query, context, payment_id)
    
    elif data == 'payment_next':
        # Show next payment
        context.user_data['current_payment_index'] += 1
        await show_payment_for_review(query, context)
    
    elif data == 'payment_prev':
        # Show previous payment
        context.user_data['current_payment_index'] -= 1
        await show_payment_for_review(query, context)
    
    elif data == 'admin_payment_settings':
        # Show payment settings
        upi_id = get_setting('upi_id', 'Not set')
        price = get_setting('premium_price', '99')
        qr_status = "✅ Set" if get_setting('qr_code_file_id') else "❌ Not set"
        
        settings_text = (
            "💰 **Payment Settings**\n\n"
            f"📱 **UPI ID:** `{upi_id}`\n"
            f"💵 **Premium Price:** ₹{price}\n"
            f"🔲 **QR Code:** {qr_status}\n\n"
            "Use these commands to update:\n"
            "`/setupi <UPI ID>`\n"
            "`/setprice <amount>`\n"
            "`/setqr` (then send QR image)"
        )
        
        keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data='admin_back')]]
        await query.edit_message_text(
            text=settings_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
    
    elif data == 'admin_premium_users':
        # Show premium users
        try:
            premium_users = db_pool.execute(
                "SELECT u.user_id, u.first_name, u.username FROM users u JOIN user_usage uu ON u.user_id = uu.user_id WHERE uu.premium_status = 1",
                fetch='all'
            )
            
            if not premium_users:
                text = "👥 **No Premium Users**\n\nNo users have premium status yet."
            else:
                text = "👥 **Premium Users**\n\n"
                for user in premium_users[:20]:
                    username = f"@{user[2]}" if user[2] else "No username"
                    text += f"• {user[1]} ({username}) - ID: `{user[0]}`\n"
                
                if len(premium_users) > 20:
                    text += f"\n... and {len(premium_users) - 20} more"
            
            keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data='admin_back')]]
            await query.edit_message_text(
                text=text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            logger.error(f"Error fetching premium users: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)}")
    
    elif data == 'admin_back':
        # Return to admin panel
        keyboard = [
            [InlineKeyboardButton("📊 User Statistics", callback_data='admin_stats')],
            [InlineKeyboardButton("📈 Tool Usage Stats", callback_data='admin_tool_stats')],
            [InlineKeyboardButton("🧾 View Pending Payments", callback_data='admin_payments')],
            [InlineKeyboardButton("👑 View Premium Users", callback_data='admin_premium_users')],
            [
                InlineKeyboardButton("✅ Grant Premium", callback_data='admin_grant_premium'),
                InlineKeyboardButton("❌ Revoke Premium", callback_data='admin_revoke_premium')
            ],
            [
                InlineKeyboardButton("🚫 Ban User", callback_data='admin_ban_user'),
                InlineKeyboardButton("✅ Unban User", callback_data='admin_unban_user')
            ],
            [InlineKeyboardButton("🎯 Set Free Limit", callback_data='admin_set_limit')],
            [InlineKeyboardButton("💰 Payment Settings", callback_data='admin_payment_settings')],
            [InlineKeyboardButton("📣 Start Broadcast", callback_data='admin_broadcast_start')],
        ]
        await query.edit_message_text(
            "👑 *Admin Panel*",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
    
    elif data == 'admin_broadcast_start':
        # Set the state for the admin user
        context.user_data['state'] = AWAITING_BROADCAST_MESSAGE
        cancel_keyboard = [[InlineKeyboardButton("❌ Cancel Broadcast", callback_data='admin_broadcast_cancel')]]
        await query.edit_message_text(
            text="📣 **Broadcast Mode**\n\nPlease send the message you want to broadcast to all users now. It can be text, photo, or a document.",
            reply_markup=InlineKeyboardMarkup(cancel_keyboard),
            parse_mode=ParseMode.MARKDOWN
        )

    elif data == 'admin_broadcast_cancel':
        # Clear the state
        context.user_data.pop('state', None)
        await query.edit_message_text(text="Broadcast canceled. Returning to the main menu.")
        # Optionally, show the main user menu again
        await query.message.reply_text("Main menu:", reply_markup=get_main_menu_keyboard())
    
    elif data == 'admin_tool_stats':
        # Show tool usage statistics
        try:
            stats = db_pool.execute(
                """SELECT tool_name, COUNT(*) as count 
                   FROM tool_usage 
                   GROUP BY tool_name 
                   ORDER BY count DESC""",
                fetch='all'
            )
            
            if not stats:
                text = "📈 **Tool Usage Stats**\n\nNo tool usage recorded yet."
            else:
                text = "📈 **Tool Usage Statistics**\n\n"
                total = sum(s[1] for s in stats)
                text += f"🔢 **Total Uses:** {total}\n\n"
                
                for tool, count in stats:
                    percentage = (count / total * 100) if total > 0 else 0
                    bar = "█" * int(percentage / 5)
                    text += f"• {tool}: {count} ({percentage:.1f}%)\n{bar}\n"
            
            keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data='admin_back')]]
            await query.edit_message_text(
                text=text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            logger.error(f"Error fetching tool stats: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)}")
    
    elif data == 'admin_grant_premium':
        # Ask for user ID to grant premium
        session_manager.set_state(ADMIN_ID, 'awaiting_grant_premium_id')
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='admin_back')]]
        await query.edit_message_text(
            "✅ **Grant Premium**\n\n"
            "Send the User ID to grant premium access.\n\n"
            "💡 **Example:** `123456789`\n\n"
            "📝 You can find User IDs in:\n"
            "• User Statistics\n"
            "• Premium Users list\n"
            "• Payment requests",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
    
    elif data == 'admin_revoke_premium':
        # Ask for user ID to revoke premium
        session_manager.set_state(ADMIN_ID, 'awaiting_revoke_premium_id')
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='admin_back')]]
        await query.edit_message_text(
            "❌ **Revoke Premium**\n\n"
            "Send the User ID to revoke premium access.\n\n"
            "💡 **Example:** `123456789`",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
    
    elif data == 'admin_ban_user':
        # Ask for user ID to ban
        session_manager.set_state(ADMIN_ID, 'awaiting_ban_user_id')
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='admin_back')]]
        await query.edit_message_text(
            "🚫 **Ban User**\n\n"
            "Send the User ID to ban from the bot.\n\n"
            "💡 **Example:** `123456789`\n\n"
            "⚠️ Banned users cannot use the bot.",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
    
    elif data == 'admin_unban_user':
        # Ask for user ID to unban
        session_manager.set_state(ADMIN_ID, 'awaiting_unban_user_id')
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='admin_back')]]
        await query.edit_message_text(
            "✅ **Unban User**\n\n"
            "Send the User ID to unban.\n\n"
            "💡 **Example:** `123456789`",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
    
    elif data == 'admin_set_limit':
        # Ask for new free limit
        session_manager.set_state(ADMIN_ID, 'awaiting_free_limit')
        current_limit = get_setting('free_usage_limit', '5')
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='admin_back')]]
        await query.edit_message_text(
            "🎯 **Set Free User Daily Limit**\n\n"
            f"📊 **Current Limit:** {current_limit} uses per day\n\n"
            "Send the new limit number.\n\n"
            "💡 **Example:** `10` (for 10 uses per day)\n"
            "💡 **Example:** `0` (to disable free usage)",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )

async def handle_broadcast_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """This function handles text messages for various user states."""
    
    user_id = update.effective_user.id
    
    # Check if user is sending a feature request
    if session_manager.get_state(user_id) == 'awaiting_feature_request':
        session_manager.clear_state(user_id)
        feature_text = update.message.text.strip()
        
        # Send to admin
        try:
            admin_message = (
                f"💡 **New Feature Request**\n\n"
                f"👤 **From:** {update.effective_user.first_name}\n"
                f"🆔 **User ID:** `{user_id}`\n"
                f"📱 **Username:** @{update.effective_user.username or 'N/A'}\n"
                f"⏰ **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"📝 **Request:**\n{feature_text}\n\n"
                f"━━━━━━━━━━━━━━━━━━━━"
            )
            await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=admin_message,
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Confirm to user
            await update.message.reply_text(
                "✅ **Feature Request Submitted!**\n\n"
                "🎉 Thank you for your feedback!\n\n"
                "Your request has been sent to our team.\n"
                "We'll review it and consider adding it in future updates.\n\n"
                "💡 Keep the suggestions coming!",
                reply_markup=get_main_menu_keyboard(),
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            logger.error(f"Failed to send feature request to admin: {e}")
            await update.message.reply_text(
                "❌ Failed to submit request. Please try again later.",
                reply_markup=get_main_menu_keyboard()
            )
        return
    
    # Check if user is waiting to send a YouTube URL
    if context.user_data.get('awaiting_ytdl_url'):
        context.user_data.pop('awaiting_ytdl_url', None)
        url = update.message.text.strip()
        
        # Enhanced validation - support more video platforms
        supported_platforms = ['youtube.com', 'youtu.be', 'tiktok.com', 'instagram.com', 
                               'facebook.com', 'twitter.com', 'x.com', 'dailymotion.com',
                               'vimeo.com', 'twitch.tv']
        
        is_valid = any(platform in url.lower() for platform in supported_platforms)
        
        if not is_valid:
            await update.message.reply_text(
                "❌ **Invalid URL!**\n\n"
                "Please send a valid video URL.\n\n"
                "🌐 **Supported Sites:**\n"
                "• YouTube, TikTok, Instagram\n"
                "• Facebook, Twitter/X\n"
                "• Vimeo, Dailymotion\n"
                "• And 1000+ more sites!\n\n"
                "💡 **Example:**\n"
                "`https://youtu.be/xxxxx`",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        # Store URL to avoid callback_data length limit (64 bytes)
        user_id = update.effective_user.id
        store_download_url(user_id, url)
        
        # Enhanced quality selection without URL in callback_data
        keyboard = [
            [
                InlineKeyboardButton("🎵 Audio (MP3)", callback_data="ytdl_audio"),
                InlineKeyboardButton("🎶 Audio (Best)", callback_data="ytdl_audio_hq")
            ],
            [
                InlineKeyboardButton("📱 Video 360p", callback_data="ytdl_360"),
                InlineKeyboardButton("📺 Video 480p", callback_data="ytdl_480")
            ],
            [
                InlineKeyboardButton("🎬 Video 720p", callback_data="ytdl_720"),
                InlineKeyboardButton("🌟 Best Quality", callback_data="ytdl_best")
            ],
            [InlineKeyboardButton("❌ Cancel", callback_data='main_menu')]
        ]
        await update.message.reply_text(
            '📺 **Advanced Video Downloader**\n\n'
            '🎯 **Choose your format:**\n\n'
            '🎵 **Audio MP3** - Fast, ~3-5MB\n'
            '🎶 **Audio Best** - High quality\n'
            '📱 **360p** - Small, ~20-40MB\n'
            '📺 **480p** - Good quality, ~40-80MB\n'
            '🎬 **720p** - HD quality, ~80-150MB\n'
            '🌟 **Best** - Highest quality available\n\n'
            '✨ **Powered by yt-dlp**\n'
            '🌐 Supports 1000+ websites!',
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Check if user is waiting to send an Instagram URL
    if context.user_data.get('awaiting_igdl_url'):
        context.user_data.pop('awaiting_igdl_url', None)
        url = update.message.text.strip()
        
        # Validate Instagram URL
        if 'instagram.com' not in url:
            await update.message.reply_text(
                "❌ **Invalid URL!**\n\n"
                "Please send a valid Instagram URL.\n\n"
                "💡 **Examples:**\n"
                "`https://www.instagram.com/p/xxxxx`\n"
                "`https://www.instagram.com/reel/xxxxx`",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        # Start download
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='main_menu')]]
        await update.message.reply_text(
            '📸 **Instagram Downloader**\n\n⏳ Starting download...',
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Download using social media download function
        await download_social_media(update, context, url, 'Instagram')
        return
    
    # Check if user is waiting to send a Facebook URL
    if context.user_data.get('awaiting_fbdl_url'):
        context.user_data.pop('awaiting_fbdl_url', None)
        url = update.message.text.strip()
        
        # Validate Facebook URL
        if 'facebook.com' not in url and 'fb.watch' not in url:
            await update.message.reply_text(
                "❌ **Invalid URL!**\n\n"
                "Please send a valid Facebook URL.\n\n"
                "💡 **Examples:**\n"
                "`https://www.facebook.com/watch/?v=xxxxx`\n"
                "`https://fb.watch/xxxxx`",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        # Start download
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='main_menu')]]
        await update.message.reply_text(
            '📘 **Facebook Downloader**\n\n⏳ Starting download...',
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Download using social media download function
        await download_social_media(update, context, url, 'Facebook')
        return
    
    # Check if user is waiting to add a note
    if context.user_data.get('awaiting_note'):
        context.user_data.pop('awaiting_note', None)
        user_id = update.effective_user.id
        note_text = update.message.text.strip()
        
        loading_msg = await update.message.reply_text("💾 Saving note...")
        with sqlite3.connect('bot_database.db') as conn:
            conn.cursor().execute("INSERT INTO notes (user_id, note) VALUES (?, ?)", (user_id, note_text))
            conn.commit()
        await loading_msg.edit_text("✅ Note saved successfully!")
        return
    
    # Check if user is waiting to send text for TTS
    if context.user_data.get('awaiting_tts'):
        context.user_data.pop('awaiting_tts', None)
        text = update.message.text.strip()
        
        loading_msg = await update.message.reply_text("🔊 Converting text to speech...")
        try:
            buf = io.BytesIO()
            tts = gTTS(text=text, lang='en')
            tts.write_to_fp(buf)
            buf.seek(0)
            await loading_msg.edit_text("📤 Sending voice message...")
            await update.message.reply_voice(voice=buf, caption="✅ Text converted to speech!")
            await loading_msg.delete()
        except Exception as e:
            await loading_msg.edit_text(f"❌ Failed to convert text to speech. Error: {str(e)}")
        return
    
    # Check if user is submitting payment reference number
    user_id = update.effective_user.id
    if session_manager.get_state(user_id) == 'awaiting_payment_reference':
        reference_number = update.message.text.strip()
        session = session_manager.get_session(user_id)
        screenshot_file_id = session['data'].get('screenshot_file_id')
        
        if not screenshot_file_id:
            await update.message.reply_text("❌ Screenshot not found. Please start again with /premium")
            session_manager.clear_state(user_id)
            return
        
        # Save to pending payments
        user = update.effective_user
        try:
            db_pool.execute(
                "INSERT INTO pending_payments (user_id, username, first_name, reference_number, screenshot_file_id) VALUES (?, ?, ?, ?, ?)",
                (user_id, user.username, user.first_name, reference_number, screenshot_file_id),
                fetch='none'
            )
            
            await update.message.reply_text(
                "✅ **Payment Submitted!**\n\n"
                "Your payment is under review.\n"
                "You'll be notified once it's verified.\n\n"
                "⏱️ Usually takes 1 hour or less.\n\n"
                "Thank you! 🙏",
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Notify admin
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_ID,
                    text=(
                        f"🔔 **New Payment Submission!**\n\n"
                        f"👤 User: {user.first_name}\n"
                        f"🆔 ID: `{user_id}`\n"
                        f"🔢 Reference: `{reference_number}`\n\n"
                        f"Use /admin to review."
                    ),
                    parse_mode=ParseMode.MARKDOWN
                )
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error saving payment: {e}")
            await update.message.reply_text(f"❌ Error submitting payment. Please try again later.")
        
        session_manager.clear_state(user_id)
        return
    
    # Admin handlers for various states
    if user_id == ADMIN_ID:
        admin_state = session_manager.get_state(ADMIN_ID)
        
        if admin_state == 'awaiting_grant_premium_id':
            try:
                target_user_id = int(update.message.text.strip())
                # Grant premium
                db_pool.execute(
                    "INSERT OR REPLACE INTO user_usage (user_id, premium_status, last_reset) VALUES (?, 1, date('now'))",
                    (target_user_id,),
                    fetch='none'
                )
                session_manager.clear_state(ADMIN_ID)
                
                await update.message.reply_text(
                    f"✅ **Premium Granted!**\n\n"
                    f"User ID: `{target_user_id}`\n"
                    f"Status: Premium Active 💎\n\n"
                    f"The user now has unlimited access!",
                    parse_mode=ParseMode.MARKDOWN
                )
                
                # Notify the user
                try:
                    await context.bot.send_message(
                        chat_id=target_user_id,
                        text="🎉 **Congratulations!**\n\n"
                             "You now have **Premium Access**! 💎\n\n"
                             "✨ Enjoy unlimited usage of all features!\n\n"
                             "Thank you for your support! 🙏",
                        parse_mode=ParseMode.MARKDOWN
                    )
                except:
                    pass
                
            except ValueError:
                await update.message.reply_text("❌ Invalid User ID. Please send a number.")
            except Exception as e:
                logger.error(f"Error granting premium: {e}")
                await update.message.reply_text(f"❌ Error: {str(e)}")
            return
        
        elif admin_state == 'awaiting_revoke_premium_id':
            try:
                target_user_id = int(update.message.text.strip())
                # Revoke premium
                db_pool.execute(
                    "UPDATE user_usage SET premium_status = 0 WHERE user_id = ?",
                    (target_user_id,),
                    fetch='none'
                )
                session_manager.clear_state(ADMIN_ID)
                
                await update.message.reply_text(
                    f"❌ **Premium Revoked**\n\n"
                    f"User ID: `{target_user_id}`\n"
                    f"Status: Free User\n\n"
                    f"The user is now subject to free tier limits.",
                    parse_mode=ParseMode.MARKDOWN
                )
                
                # Notify the user
                try:
                    await context.bot.send_message(
                        chat_id=target_user_id,
                        text="📢 **Premium Status Update**\n\n"
                             "Your premium access has been revoked.\n\n"
                             "You are now on the free tier with daily limits.\n\n"
                             "To regain premium, use /premium",
                        parse_mode=ParseMode.MARKDOWN
                    )
                except:
                    pass
                
            except ValueError:
                await update.message.reply_text("❌ Invalid User ID. Please send a number.")
            except Exception as e:
                logger.error(f"Error revoking premium: {e}")
                await update.message.reply_text(f"❌ Error: {str(e)}")
            return
        
        elif admin_state == 'awaiting_ban_user_id':
            try:
                target_user_id = int(update.message.text.strip())
                # Ban user
                db_pool.execute(
                    "INSERT OR REPLACE INTO banned_users (user_id, banned_at) VALUES (?, datetime('now'))",
                    (target_user_id,),
                    fetch='none'
                )
                session_manager.clear_state(ADMIN_ID)
                
                await update.message.reply_text(
                    f"🚫 **User Banned**\n\n"
                    f"User ID: `{target_user_id}`\n"
                    f"Status: Banned\n\n"
                    f"This user can no longer use the bot.",
                    parse_mode=ParseMode.MARKDOWN
                )
                
                # Notify the user
                try:
                    await context.bot.send_message(
                        chat_id=target_user_id,
                        text="🚫 **Access Restricted**\n\n"
                             "Your access to this bot has been restricted.\n\n"
                             "Contact the administrator for more information.",
                        parse_mode=ParseMode.MARKDOWN
                    )
                except:
                    pass
                
            except ValueError:
                await update.message.reply_text("❌ Invalid User ID. Please send a number.")
            except Exception as e:
                logger.error(f"Error banning user: {e}")
                await update.message.reply_text(f"❌ Error: {str(e)}")
            return
        
        elif admin_state == 'awaiting_unban_user_id':
            try:
                target_user_id = int(update.message.text.strip())
                # Unban user
                db_pool.execute(
                    "DELETE FROM banned_users WHERE user_id = ?",
                    (target_user_id,),
                    fetch='none'
                )
                session_manager.clear_state(ADMIN_ID)
                
                await update.message.reply_text(
                    f"✅ **User Unbanned**\n\n"
                    f"User ID: `{target_user_id}`\n"
                    f"Status: Active\n\n"
                    f"The user can now use the bot again.",
                    parse_mode=ParseMode.MARKDOWN
                )
                
                # Notify the user
                try:
                    await context.bot.send_message(
                        chat_id=target_user_id,
                        text="✅ **Access Restored**\n\n"
                             "Your access to this bot has been restored.\n\n"
                             "Welcome back! Use /start to begin.",
                        parse_mode=ParseMode.MARKDOWN
                    )
                except:
                    pass
                
            except ValueError:
                await update.message.reply_text("❌ Invalid User ID. Please send a number.")
            except Exception as e:
                logger.error(f"Error unbanning user: {e}")
                await update.message.reply_text(f"❌ Error: {str(e)}")
            return
        
        elif admin_state == 'awaiting_free_limit':
            try:
                new_limit = int(update.message.text.strip())
                if new_limit < 0:
                    await update.message.reply_text("❌ Limit must be 0 or greater.")
                    return
                
                set_setting('free_usage_limit', str(new_limit))
                session_manager.clear_state(ADMIN_ID)
                
                await update.message.reply_text(
                    f"✅ **Free Limit Updated!**\n\n"
                    f"New daily limit: {new_limit} uses\n\n"
                    f"{'🔒 Free usage disabled' if new_limit == 0 else '✨ Free users can now use ' + str(new_limit) + ' tools per day'}",
                    parse_mode=ParseMode.MARKDOWN
                )
                
            except ValueError:
                await update.message.reply_text("❌ Invalid number. Please send a valid limit.")
            except Exception as e:
                logger.error(f"Error setting limit: {e}")
                await update.message.reply_text(f"❌ Error: {str(e)}")
            return
    
    # Admin broadcast handler
    # Only process if user is admin AND in broadcast state
    if update.effective_user.id != ADMIN_ID or context.user_data.get('state') != AWAITING_BROADCAST_MESSAGE:
        # This message is not a broadcast or special state, so ignore it
        return

    # Clear the state to prevent accidental re-broadcasts
    context.user_data.pop('state', None)
    
    with sqlite3.connect('bot_database.db') as conn:
        user_ids = conn.cursor().execute("SELECT user_id FROM users").fetchall()

    await update.message.reply_text(f"✅ Message received. Starting broadcast to {len(user_ids)} users. This may take a while...")
    
    success_count = 0
    fail_count = 0
    for user_id_tuple in user_ids:
        user_id = user_id_tuple[0]
        try:
            # Forward the admin's message to the user
            await context.bot.copy_message(chat_id=user_id, from_chat_id=update.message.chat_id, message_id=update.message.message_id)
            success_count += 1
        except Forbidden:
            logger.warning(f"User {user_id} has blocked the bot. Skipping.")
            fail_count += 1
        except Exception as e:
            logger.error(f"Failed to send message to {user_id}: {e}")
            fail_count += 1
        await asyncio.sleep(0.1) # Avoid hitting Telegram's rate limits

    await update.message.reply_text(f"Broadcast finished!\n\n✅ Sent successfully to: {success_count} users.\n❌ Failed for: {fail_count} users.")


# --- 6. User-Facing Functions (Keyboards, start, help, etc.) ---
# ... (All user-facing code from the previous version remains exactly the same) ...
def get_main_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("🧰 File Converter", callback_data='menu_file'),
         InlineKeyboardButton("💬 Text Tools", callback_data='menu_text')],
        [InlineKeyboardButton("📥 Downloads", callback_data='menu_downloader'),
         InlineKeyboardButton("📅 Productivity", callback_data='menu_productivity')],
        [InlineKeyboardButton("☁️ File Storage", callback_data='menu_storage'),
         InlineKeyboardButton("🔐 Security", callback_data='menu_security')],
        [InlineKeyboardButton("👤 My Account", callback_data='my_account'),
         InlineKeyboardButton("💎 Get Premium", callback_data='get_premium')],
        [InlineKeyboardButton("❤️ Donate", callback_data='donate'),
         InlineKeyboardButton("ℹ️ Help", callback_data='menu_help')],
        [InlineKeyboardButton("💡 Request New Feature", callback_data='request_feature')],
    ]
    return InlineKeyboardMarkup(keyboard)

def get_back_button_keyboard():
    keyboard = [[InlineKeyboardButton("⬅️ Back to Main Menu", callback_data='main_menu')]]
    return InlineKeyboardMarkup(keyboard)

def get_file_converter_keyboard():
    keyboard = [
        [InlineKeyboardButton("🖼️ Image Converter", callback_data='help_image'),
         InlineKeyboardButton("🗜️ Compress Image", callback_data='help_compress')],
        [InlineKeyboardButton("🎵 Video to MP3", callback_data='help_video'),
         InlineKeyboardButton("📸 Photos to PDF", callback_data='help_photo_pdf')],
        [InlineKeyboardButton("📄 PDF to DOCX", callback_data='help_pdf')],
        [InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_text_tools_keyboard():
    keyboard = [
        [InlineKeyboardButton("🔊 Text to Speech", callback_data='action_tts')],
        [InlineKeyboardButton("🌍 Translator", callback_data='action_translate')],
        [InlineKeyboardButton("🔐 Encrypt", callback_data='action_encrypt'),
         InlineKeyboardButton("🔓 Decrypt", callback_data='action_decrypt')],
        [InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_downloader_keyboard():
    keyboard = [
        [InlineKeyboardButton("📺 Video Downloader", callback_data='action_ytdl')],
        [InlineKeyboardButton("🔗 Shorten URL", callback_data='action_shorten'),
         InlineKeyboardButton("📱 QR Code", callback_data='action_qr')],
        [InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_video_downloader_back_only():
    """Keyboard with only Back button for video downloader screen"""
    keyboard = [
        [InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_productivity_keyboard():
    keyboard = [
        [InlineKeyboardButton("➕ Add Note", callback_data='action_addnote')],
        [InlineKeyboardButton("📋 View Notes", callback_data='action_viewnotes')],
        [InlineKeyboardButton("🗑️ Delete Note", callback_data='action_delnote')],
        [InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_security_keyboard():
    keyboard = [
        [InlineKeyboardButton("🔑 Generate Password", callback_data='action_genpass')],
        [InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_storage_keyboard():
    keyboard = [
        [InlineKeyboardButton("📤 Upload File", callback_data='action_upload_file')],
        [InlineKeyboardButton("📂 My Files", callback_data='action_list_files')],
        [InlineKeyboardButton("🗑️ Delete File", callback_data='action_delete_file')],
        [InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]
    ]
    return InlineKeyboardMarkup(keyboard)

async def show_loading_animation(message, text_base):
    """Shows a loading animation by editing the message"""
    animations = ["⏳", "⌛", "⏳", "⌛"]
    for i in range(4):
        try:
            await message.edit_text(f"{animations[i]} {text_base}")
            await asyncio.sleep(0.3)
        except:
            pass

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    
    # Rate limiting check
    if not rate_limiter.consume(user_id, tokens=1):
        wait_time = rate_limiter.get_wait_time(user_id)
        await update.message.reply_text(f"⏳ Please wait {wait_time:.1f} seconds before trying again.")
        return
    
    # Initialize user session
    session = session_manager.get_session(user_id)
    session_manager.clear_state(user_id)  # Clear any existing state
    
    # Store user in database using connection pool
    try:
        db_pool.execute(
            "INSERT OR IGNORE INTO users (user_id, first_name, username, verified) VALUES (?, ?, ?, 0)",
            (user_id, user.first_name, user.username),
            fetch='none'
        )
    except Exception as e:
        logger.error(f"Database error in start: {e}")
    
    # Check if user needs verification
    if not is_user_verified(user_id):
        verification_text = (
            f"👋 **Welcome, {user.first_name}!**\n\n"
            f"🤖 **Welcome to Super Bot!**\n\n"
            f"🔒 **Quick Verification Required**\n\n"
            f"To ensure you're a real user and not a bot, please click the verification button below.\n\n"
            f"✅ This is a one-time verification\n"
            f"⚡ Takes just 1 second\n\n"
            f"👇 Click the button to get started:"
        )
        keyboard = [[InlineKeyboardButton("✅ Verify & Start Using Bot", callback_data='verify_user')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(verification_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
        return
    
    welcome_text = (
        f"👋 **Welcome, {user.first_name}!**\n\n"
        f"🤖 I am your **All-In-One Utility Bot**\n\n"
        f"✨ I can help you with:\n"
        f"• File conversions (Images, Videos, PDFs)\n"
        f"• Text tools (TTS, Translation, Encryption)\n"
        f"• Downloads (YouTube, Instagram, Facebook)\n"
        f"• Productivity (Notes management)\n"
        f"• Security (Password generator)\n\n"
        f"⚡ **Optimized for multiple users!**\n"
        f"👇 Choose a category to get started:"
    )
    await update.message.reply_markdown(welcome_text, reply_markup=get_main_menu_keyboard())

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🆘 **Quick Help**\n\n"
        "Use the buttons below to explore features, or use these commands:\n\n"
        "📝 **Commands:**\n"
        "`/start` - Show main menu\n"
        "`/help` - Show this help\n"
        "`/admin` - Admin panel (admin only)\n\n"
        "💡 **Tip:** Most features work with inline buttons!"
    )
    await update.message.reply_markdown(help_text, reply_markup=get_main_menu_keyboard())

async def account_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show user account information"""
    user_id = update.effective_user.id
    stats = get_user_stats(user_id)
    
    if stats['premium']:
        account_text = (
            "👤 **My Account**\n\n"
            "✅ **Premium User**\n\n"
            "🎉 You have unlimited access to all features!\n\n"
            "💎 Status: Active\n"
            "⚡ Usage: Unlimited\n\n"
            "Thank you for your support! 🙏"
        )
    else:
        account_text = (
            "👤 **My Account**\n\n"
            "🆓 **Free User**\n\n"
            f"📊 **Today's Usage:** {stats['usage_count']} / 5\n"
            f"⏰ **Resets in:** {stats['reset_time']}\n"
            f"✨ **Remaining:** {stats['remaining']} uses\n\n"
            "💎 **Upgrade to Premium** for unlimited access!\n"
            "Tap the button below to learn more."
        )
    
    keyboard = []
    if not stats['premium']:
        keyboard.append([InlineKeyboardButton("💎 Get Premium", callback_data='get_premium')])
    keyboard.append([InlineKeyboardButton("⬅️ Main Menu", callback_data='main_menu')])
    
    await update.message.reply_text(
        account_text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN
    )

async def premium_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show premium information"""
    price = get_setting('premium_price', '99')
    
    premium_text = (
        "💎 *Premium Membership*\n\n"
        "✨ *Benefits:*\n"
        "• ♾️ Unlimited tool usage\n"
        "• ⚡ No daily limits\n"
        "• 🚀 Priority processing\n"
        "• 💝 Support development\n\n"
        f"💰 *Price:* ₹{price} (One-time)\n\n"
        "📝 *How to upgrade:*\n"
        "1. Click 'Buy Premium' below\n"
        "2. Make payment via UPI\n"
        "3. Upload screenshot\n"
        "4. Get instant activation!\n\n"
        "⏱️ Usually activated within 1 hour."
    )
    
    keyboard = [
        [InlineKeyboardButton("💎 Buy Premium Now", callback_data='get_premium')],
        [InlineKeyboardButton("⬅️ Main Menu", callback_data='main_menu')]
    ]
    
    await update.message.reply_text(
        premium_text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN
    )

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    logger.info(f"Button callback received: {query.data} from user {user_id}")
    
    # Handle verification callback
    if query.data == 'verify_user':
        if verify_user(user_id):
            await query.answer("✅ Verification successful!", show_alert=False)
            
            # Send notification to admin
            try:
                admin_message = (
                    f"🆕 **New User Verified!**\n\n"
                    f"👤 **User:** {query.from_user.first_name}\n"
                    f"🆔 **User ID:** `{user_id}`\n"
                    f"📱 **Username:** @{query.from_user.username or 'N/A'}\n"
                    f"⏰ **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    f"✅ User has been verified and can now use the bot."
                )
                await context.bot.send_message(
                    chat_id=ADMIN_ID,
                    text=admin_message,
                    parse_mode=ParseMode.MARKDOWN
                )
            except Exception as e:
                logger.error(f"Failed to send admin notification: {e}")
            
            welcome_text = (
                f"✅ **Verification Successful!**\n\n"
                f"🎉 Welcome to Super Bot, {query.from_user.first_name}!\n\n"
                f"🤖 I am your **All-In-One Utility Bot**\n\n"
                f"✨ I can help you with:\n"
                f"• File conversions (Images, Videos, PDFs)\n"
                f"• Text tools (TTS, Translation, Encryption)\n"
                f"• Downloads (YouTube, Instagram, Facebook)\n"
                f"• Productivity (Notes management)\n"
                f"• Security (Password generator)\n\n"
                f"⚡ **Optimized for multiple users!**\n"
                f"👇 Choose a category to get started:"
            )
            await query.edit_message_text(
                text=welcome_text,
                reply_markup=get_main_menu_keyboard(),
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await query.answer("❌ Verification failed. Please try again.", show_alert=True)
        return
    
    # Check if user is verified before allowing any other actions
    if not is_user_verified(user_id):
        await query.answer(
            "🔒 Please complete verification first by using /start command",
            show_alert=True
        )
        return
    
    # Check if user is busy with another task
    if task_lock.is_user_busy(user_id):
        active_task = task_lock.get_active_task(user_id)
        await query.answer(
            f"⏳ Please wait! You have an active task: {active_task}",
            show_alert=True
        )
        return
    
    try:
        await query.answer()
    except Exception as e:
        # Query might be too old, ignore error
        logger.warning(f"Could not answer callback query: {e}")
        pass
    
    data = query.data
    text, keyboard = "", get_back_button_keyboard()
    
    if data == 'main_menu':
        logger.info(f"Main menu requested by user {user_id}")
        text = "👇 *Choose a category:*"
        keyboard = get_main_menu_keyboard()
    
    elif data == 'menu_file':
        text = (
            "🧰 *File Converter Suite*\n\n"
            "Convert your files instantly!\n\n"
            "📸 *Image Converter*\n"
            "• Send any photo to convert to PNG & WEBP\n\n"
            "🎵 *Video to MP3*\n"
            "• Send a video to extract audio as MP3\n\n"
            "📄 *PDF to DOCX*\n"
            "• Send a PDF to convert to Word document\n\n"
            "💡 *Just send the file to start!*"
        )
        keyboard = get_file_converter_keyboard()
    
    elif data == 'menu_text':
        text = (
            "💬 *Text Tools*\n\n"
            "Powerful text manipulation tools:\n\n"
            "🔊 *Text to Speech:* `/tts [text]`\n"
            "🌍 *Translator:* `/translate [lang] [text]`\n"
            "🔐 *Encrypt:* `/encrypt [password] [text]`\n"
            "🔓 *Decrypt:* `/decrypt [password] [text]`\n\n"
            "📝 *Examples:*\n"
            "`/tts Hello World`\n"
            "`/translate es Hello`"
        )
        keyboard = get_text_tools_keyboard()
    
    elif data == 'menu_downloader':
        text = (
            "📥 *Downloader & Link Tools*\n\n"
            "📺 *Advanced Video Downloader*\n"
            "`/ytdl [URL]` - Download from 1000+ sites\n"
            "🌐 YouTube, TikTok, Instagram, Facebook, Twitter & more!\n\n"
            "🔗 *URL Shortener*\n"
            "`/shorten [URL]` - Shorten any link\n\n"
            "📱 *QR Code Generator*\n"
            "`/qr [text/URL]` - Create QR code\n\n"
            "💡 *Quality Options:*\n"
            "🎵 Audio (MP3/Best) | 🎬 Video (360p-720p-Best)\n\n"
            "💡 *Example:*\n"
            "`/ytdl https://youtu.be/xxxxx`"
        )
        keyboard = get_downloader_keyboard()
    
    elif data == 'menu_productivity':
        text = (
            "📅 *Productivity Tools*\n\n"
            "Keep track of your notes:\n\n"
            "➕ *Add Note:* `/addnote [your note]`\n"
            "📋 *View Notes:* `/notes`\n"
            "🗑️ *Delete Note:* `/delnote [id]`\n\n"
            "💡 *Example:*\n"
            "`/addnote Buy groceries tomorrow`"
        )
        keyboard = get_productivity_keyboard()
    
    elif data == 'menu_security':
        text = (
            "🔐 *Security & Utility*\n\n"
            "🔑 *Password Generator*\n"
            "Generate secure passwords:\n\n"
            "`/password` - Generate 16-char password\n"
            "`/password [length]` - Custom length (8-128)\n\n"
            "💡 *Example:*\n"
            "`/password 20`"
        )
        keyboard = get_security_keyboard()
    
    elif data == 'menu_storage':
        text = (
            "☁️ *File Storage & Management*\n\n"
            "Store your files securely on Google Drive and access them anytime!\n\n"
            "📤 *Upload File:* Send any file to store\n"
            "📂 *My Files:* View all your stored files\n"
            "🗑️ *Delete File:* Remove files from storage\n\n"
            "✨ *Features:*\n"
            "• Unlimited storage (uses your Google Drive)\n"
            "• Permanent shareable links\n"
            "• Access files anytime, anywhere\n"
            "• Secure cloud storage\n\n"
            "💡 *How to use:*\n"
            "1. Click 'Upload File' and send any file\n"
            "2. Get a permanent Google Drive link\n"
            "3. Share or access your files anytime!"
        )
        keyboard = get_storage_keyboard()
    
    elif data == 'menu_help':
        text = (
            "ℹ️ *Help & Information*\n\n"
            "*How to use this bot:*\n\n"
            "1️⃣ Choose a category from the main menu\n"
            "2️⃣ Follow the instructions for each tool\n"
            "3️⃣ Send files or use commands\n\n"
            "*File Conversions:*\n"
            "Just send the file directly (photo/video/PDF)\n\n"
            "*Text Commands:*\n"
            "Type the command followed by your input\n\n"
            "*Need help?* Contact support or check examples in each category!"
        )
        keyboard = get_back_button_keyboard()
    
    # Individual help pages
    elif data == 'help_image':
        text = "🖼️ *Image Converter*\n\n📸 Simply send any photo\n✅ You'll get PNG and WEBP formats\n⚡ Fast conversion!"
        keyboard = get_file_converter_keyboard()
    
    elif data == 'help_video':
        text = "🎵 **Video to MP3**\n\n📤 Send a video file\n✅ Audio will be extracted as MP3\n⚡ High quality output!"
        keyboard = get_file_converter_keyboard()
    
    elif data == 'help_pdf':
        text = "📄 **PDF to DOCX**\n\n📤 Send a PDF document\n✅ Converted to editable Word format\n⚡ Preserves text content!"
        keyboard = get_file_converter_keyboard()
    
    elif data == 'help_compress':
        # Show quality selection buttons
        text = (
            "🗜️ **Image Compressor**\n\n"
            "📸 **Choose compression quality:**\n\n"
            "🟢 **High (85%)** - Best quality, moderate compression\n"
            "🟡 **Medium (70%)** - Balanced quality & size\n"
            "� **Low (50%)** - Smallest file size\n\n"
            "💡 Select a quality level below:"
        )
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🟢 High Quality (85%)", callback_data='compress_high')],
            [InlineKeyboardButton("🟡 Medium Quality (70%)", callback_data='compress_medium')],
            [InlineKeyboardButton("🔴 Low Quality (50%)", callback_data='compress_low')],
            [InlineKeyboardButton("⬅️ Back", callback_data='menu_file')]
        ])
    
    elif data == 'help_photo_pdf':
        # Start photo collection for PDF
        user_id = query.from_user.id
        session_manager.set_state(user_id, 'collecting_photos_for_pdf')
        context.user_data['pdf_photos'] = []
        
        logger.info(f"PHOTO PDF: Started collecting for user {user_id}. State set to: collecting_photos_for_pdf")
        
        text = (
            "📸 **Photos to PDF**\n\n"
            "📚 **Started collecting photos!**\n\n"
            "📤 Send me all your photos one by one\n"
            "✅ 'Done Sending' button will appear after you send photos\n\n"
            "💡 All photos will be combined into a single PDF in the order you send them.\n\n"
            "✨ **Features:**\n"
            "• Unlimited photos\n"
            "• High-quality PDF\n"
            "• Auto page sizing (A4)"
        )
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("❌ Cancel", callback_data='menu_file')]
        ])
    
    elif data == 'help_tts':
        text = "🔊 **Text to Speech**\n\n📝 Usage: `/tts Your text here`\n✅ Generates voice message\n🎙️ Natural English voice!"
        keyboard = get_text_tools_keyboard()
    
    elif data == 'help_translate':
        text = "🌍 **Translator**\n\n📝 Usage: `/translate [lang] [text]`\n✅ Example: `/translate es Hello`\n🌐 Supports 100+ languages!"
        keyboard = get_text_tools_keyboard()
    
    elif data == 'help_encrypt':
        text = "🔐 **Encrypt Text**\n\n📝 Usage: `/encrypt [pass] [text]`\n✅ Example: `/encrypt mypass Secret`\n🔒 Military-grade encryption!"
        keyboard = get_text_tools_keyboard()
    
    elif data == 'help_decrypt':
        text = "🔓 **Decrypt Text**\n\n📝 Usage: `/decrypt [pass] [encrypted]`\n✅ Use same password as encryption\n🔑 Secure decryption!"
        keyboard = get_text_tools_keyboard()
    
    elif data == 'help_ytdl':
        text = "📺 **YouTube Downloader**\n\n📝 Usage: `/ytdl [URL]`\n✅ Choose video or audio format\n⚡ Fast download!"
        keyboard = get_downloader_keyboard()
    
    elif data == 'help_shorten':
        text = "🔗 **URL Shortener**\n\n📝 Usage: `/shorten [URL]`\n✅ Example: `/shorten https://example.com`\n⚡ Instant short link!"
        keyboard = get_downloader_keyboard()
    
    elif data == 'help_qr':
        text = "📱 **QR Code Generator**\n\n📝 Usage: `/qr [text or URL]`\n✅ Example: `/qr https://example.com`\n📷 High-quality QR code!"
        keyboard = get_downloader_keyboard()
    
    elif data == 'help_addnote':
        text = "➕ **Add Note**\n\n📝 Usage: `/addnote [your note]`\n✅ Example: `/addnote Meeting at 3pm`\n💾 Saved to your notes!"
        keyboard = get_productivity_keyboard()
    
    elif data == 'help_notes':
        text = "📋 **View Notes**\n\n📝 Usage: `/notes`\n✅ Shows all your saved notes\n📅 With dates and IDs!"
        keyboard = get_productivity_keyboard()
    
    elif data == 'help_delnote':
        text = "🗑️ **Delete Note**\n\n📝 Usage: `/delnote [id]`\n✅ Example: `/delnote 5`\n💡 Use `/notes` to see IDs!"
        keyboard = get_productivity_keyboard()
    
    elif data == 'help_password':
        text = "🔑 **Password Generator**\n\n📝 Usage: `/password [length]`\n✅ Example: `/password 20`\n🔐 Cryptographically secure!"
        keyboard = get_security_keyboard()
    
    # Premium and Account handlers
    elif data == 'get_premium':
        # Show payment instructions
        logger.info(f"Get premium button clicked by user {query.from_user.id}")
        
        upi_id = get_setting('upi_id', 'Not configured')
        price = get_setting('premium_price', '99')
        qr_file_id = get_setting('qr_code_file_id')
        
        payment_text = (
            "💎 *Get Premium Access*\n\n"
            f"💰 *Amount:* ₹{price}\n\n"
            f"📱 *UPI ID:* `{upi_id}`\n\n"
            "📝 *Payment Steps:*\n"
            "1. Pay ₹{} to the UPI ID above\n"
            "2. Take a screenshot of payment\n"
            "3. Click 'Submit Payment' below\n"
            "4. Upload screenshot and reference number\n\n"
            "⏱️ Activation within 1 hour!"
        ).format(price)
        
        keyboard = [
            [InlineKeyboardButton("📤 Submit Payment", callback_data='submit_payment')],
            [InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]
        ]
        
        # Send QR code if available
        if qr_file_id:
            try:
                # Delete old message first
                await query.message.delete()
                
                # Send new message with QR code
                await context.bot.send_photo(
                    chat_id=query.message.chat_id,
                    photo=qr_file_id,
                    caption=payment_text,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode=ParseMode.MARKDOWN
                )
                logger.info(f"Premium info with QR sent to user {query.from_user.id}")
                return
            except Exception as e:
                logger.error(f"Error sending QR code: {e}")
                # If QR sending fails, fall through to text-only version
                pass
        
        # No QR code or QR sending failed - send text only
        await query.edit_message_text(
            text=payment_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
        logger.info(f"Premium info (text only) sent to user {query.from_user.id}")
        return
    
    elif data == 'donate':
        # Show donation information
        upi_id = get_setting('upi_id', 'Not configured')
        qr_file_id = get_setting('qr_code_file_id')
        
        donation_text = (
            "❤️ **Support Our Bot**\n\n"
            "Thank you for considering a donation! Your support helps us:\n"
            "• Keep the bot running 24/7\n"
            "• Add new features\n"
            "• Improve performance\n"
            "• Maintain server costs\n\n"
            f"💰 **Donate via UPI:** `{upi_id}`\n\n"
            "🙏 Every contribution, big or small, is greatly appreciated!\n\n"
            "💡 **Note:** Donations are voluntary and don't provide premium features.\n"
            "For unlimited access, please use the Premium option."
        )
        
        keyboard = [
            [InlineKeyboardButton("💎 Get Premium Instead", callback_data='get_premium')],
            [InlineKeyboardButton("⬅️ Back to Menu", callback_data='main_menu')]
        ]
        
        # Send QR code if available
        if qr_file_id:
            try:
                await query.message.reply_photo(
                    photo=qr_file_id,
                    caption=donation_text,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode=ParseMode.MARKDOWN
                )
                await query.delete_message()
                return
            except:
                pass
        
        await query.edit_message_text(
            text=donation_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    elif data == 'request_feature':
        # Feature request handler
        user_id = query.from_user.id
        session_manager.set_state(user_id, 'awaiting_feature_request')
        
        text = (
            "💡 **Request New Feature**\n\n"
            "🎯 **We love hearing from you!**\n\n"
            "Have an idea for a new tool or feature?\n"
            "Want to see something added to the bot?\n\n"
            "📝 **Please describe your feature request:**\n\n"
            "💡 **Examples:**\n"
            "• Add voice message transcription\n"
            "• Support for more video platforms\n"
            "• Background removal from images\n"
            "• Custom watermarks on videos\n\n"
            "✍️ **Just type your request now!**\n\n"
            "Your feedback helps us improve! 🚀"
        )
        
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data='main_menu')]])
    
    elif data == 'submit_payment':
        # Ask user to send screenshot and reference
        user_id = query.from_user.id
        logger.info(f"Submit payment button clicked by user {user_id}")
        
        try:
            session_manager.set_state(user_id, 'awaiting_payment_screenshot')
            
            text = (
                "📤 **Submit Payment Proof**\n\n"
                "Please send:\n"
                "1️⃣ Screenshot of payment (as image)\n\n"
                "After sending the screenshot, I'll ask for your UPI Reference Number."
            )
            
            keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data='main_menu')]]
            await query.edit_message_text(
                text=text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.MARKDOWN
            )
            logger.info(f"Submit payment message sent successfully to user {user_id}")
        except Exception as e:
            logger.error(f"Error in submit_payment handler: {e}", exc_info=True)
            await query.answer("❌ Error processing request. Please try again.", show_alert=True)
        return
    
    elif data == 'my_account':
        # Same as /account command
        user_id = query.from_user.id
        stats = get_user_stats(user_id)
        
        if stats['premium']:
            text = (
                "👤 **My Account**\n\n"
                "✅ **Premium User**\n\n"
                "🎉 You have unlimited access to all features!\n\n"
                "💎 Status: Active\n"
                "⚡ Usage: Unlimited\n\n"
                "Thank you for your support! 🙏"
            )
        else:
            text = (
                "👤 **My Account**\n\n"
                "🆓 **Free User**\n\n"
                f"📊 **Today's Usage:** {stats['usage_count']} / 5\n"
                f"⏰ **Resets in:** {stats['reset_time']}\n"
                f"✨ **Remaining:** {stats['remaining']} uses\n\n"
                "💎 **Upgrade to Premium** for unlimited access!"
            )
        
        buttons = []
        if not stats['premium']:
            buttons.append([InlineKeyboardButton("💎 Get Premium", callback_data='get_premium')])
        buttons.append([InlineKeyboardButton("⬅️ Back", callback_data='main_menu')])
        
        await query.edit_message_text(
            text=text,
            reply_markup=InlineKeyboardMarkup(buttons),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Action handlers - Actually perform the actions!
    elif data == 'action_genpass':
        # Generate a random password
        length = 16
        alphabet = string.ascii_letters + string.digits + string.punctuation
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        text = (
            f"✅ **Secure Password Generated!**\n\n"
            f"🔐 `{password}`\n\n"
            f"📏 Length: {length} characters\n"
            f"💡 Tap to copy!\n\n"
            f"🔄 Want a different length? Use `/password [8-128]`"
        )
        keyboard = get_security_keyboard()
    
    elif data == 'action_viewnotes':
        # View all notes for the user - using connection pool
        user_id = query.from_user.id
        try:
            notes = db_pool.execute(
                "SELECT id, note, created_at FROM notes WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
                fetch='all'
            )
        except Exception as e:
            logger.error(f"Database error fetching notes: {e}")
            notes = []
        
        if not notes:
            text = "📝 You have no saved notes.\n\n💡 Use the **➕ Add Note** button to create one!"
        else:
            text = "📋 **Your Notes:**\n\n"
            for note in notes[:10]:  # Show max 10 notes
                note_id, note_text, created_at = note[0], note[1], note[2]
                try:
                    date = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S').strftime('%d %b')
                except:
                    date = "Recent"
                text += f"🔹 **ID:** `{note_id}` | 📅 {date}\n📝 {note_text[:50]}{'...' if len(note_text) > 50 else ''}\n\n"
            text += "💡 Use `/delnote [id]` to delete a note"
        keyboard = get_productivity_keyboard()
    
    elif data == 'action_addnote':
        user_id = query.from_user.id
        session_manager.set_state(user_id, 'awaiting_note')
        text = (
            "➕ **Add a New Note**\n\n"
            "📝 Please send your note text now.\n\n"
            "💡 **Example:**\n"
            "Just type: `Buy groceries tomorrow`\n\n"
            "Or use the command:\n"
            "`/addnote Your note here`"
        )
        keyboard = get_productivity_keyboard()
        context.user_data['awaiting_note'] = True  # Keep for backward compatibility
    
    elif data == 'action_delnote':
        text = (
            "🗑️ **Delete a Note**\n\n"
            "First, view your notes to see the IDs,\n"
            "then use:\n\n"
            "`/delnote [id]`\n\n"
            "💡 **Example:** `/delnote 5`"
        )
        keyboard = get_productivity_keyboard()
    
    elif data == 'action_upload_file':
        user_id = query.from_user.id
        session_manager.set_state(user_id, 'awaiting_file_upload')
        text = (
            "📤 **Upload File to Storage**\n\n"
            "📎 **Please send me any file:**\n\n"
            "✨ **What happens:**\n"
            "1. File uploads to Google Drive\n"
            "2. You get a permanent shareable link\n"
            "3. Access it anytime, anywhere\n\n"
            "📊 **Supported:**\n"
            "• Documents (PDF, DOC, TXT, etc.)\n"
            "• Images (JPG, PNG, WebP, etc.)\n"
            "• Videos (MP4, MKV, AVI, etc.)\n"
            "• Archives (ZIP, RAR, 7Z, etc.)\n"
            "• Any file type!\n\n"
            "💡 **Just drag and drop or send your file now!**"
        )
        keyboard = get_storage_keyboard()
        context.user_data['awaiting_file_upload'] = True
    
    elif data == 'action_list_files':
        user_id = query.from_user.id
        result = db_pool.execute(
            "SELECT id, filename, file_size, share_link, uploaded_at FROM stored_files WHERE user_id = ? ORDER BY uploaded_at DESC",
            (user_id,)
        )
        
        if not result:
            text = (
                "📂 **My Files**\n\n"
                "📭 You haven't uploaded any files yet.\n\n"
                "💡 Click 'Upload File' to store your first file!"
            )
        else:
            text = "📂 **My Files**\n\n"
            for file_id, filename, file_size, share_link, uploaded_at in result:
                # Format file size
                size_mb = file_size / (1024 * 1024)
                if size_mb < 1:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{size_mb:.1f} MB"
                
                # Format date
                try:
                    date = datetime.strptime(uploaded_at, '%Y-%m-%d %H:%M:%S').strftime('%d %b %Y')
                except:
                    date = "Recent"
                
                text += f"📄 **{filename}**\n"
                text += f"   📊 Size: {size_str}\n"
                text += f"   📅 Uploaded: {date}\n"
                text += f"   🔗 [Open File]({share_link})\n"
                text += f"   🆔 ID: `{file_id}`\n\n"
            
            text += f"📊 **Total Files:** {len(result)}\n\n"
            text += "💡 Use 'Delete File' to remove files by ID"
        
        # Dynamic keyboard: include per-file delete buttons (max 5 to avoid clutter)
        file_buttons = []
        if result:
            for file_row in result[:5]:  # Show delete buttons for first 5 files
                fid = file_row[0]
                file_buttons.append([InlineKeyboardButton(f"🗑️ Delete ID {fid}", callback_data=f"delete_stored_{fid}")])
        file_buttons.append([InlineKeyboardButton("📤 Upload", callback_data='action_upload_file'), InlineKeyboardButton("🔄 Refresh", callback_data='action_list_files')])
        file_buttons.append([InlineKeyboardButton("⬅️ Back", callback_data='menu_storage')])
        keyboard = InlineKeyboardMarkup(file_buttons)
    
    elif data == 'action_delete_file':
        user_id = query.from_user.id
        result = db_pool.execute(
            "SELECT id, filename FROM stored_files WHERE user_id = ? ORDER BY uploaded_at DESC",
            (user_id,)
        )
        
        if not result:
            text = (
                "🗑️ **Delete File**\n\n"
                "📭 You don't have any files to delete.\n\n"
                "💡 Upload files first using 'Upload File'!"
            )
        else:
            text = "🗑️ **Delete File**\n\n"
            text += "📋 **Your Files:**\n\n"
            for file_id, filename in result:
                text += f"🔹 **ID:** `{file_id}` - {filename}\n"
            
            text += f"\n💡 **To delete a file, use:**\n"
            text += "`/deletefile [id]`\n\n"
            text += "**Example:** `/deletefile 5`"
        
        # Provide delete buttons for quick access
        del_buttons = []
        if result:
            for fid, fname in result[:5]:
                del_buttons.append([InlineKeyboardButton(f"🗑️ Delete {fid}", callback_data=f"delete_stored_{fid}")])
        del_buttons.append([InlineKeyboardButton("📂 My Files", callback_data='action_list_files'), InlineKeyboardButton("📤 Upload", callback_data='action_upload_file')])
        del_buttons.append([InlineKeyboardButton("⬅️ Back", callback_data='menu_storage')])
        keyboard = InlineKeyboardMarkup(del_buttons)
    
    elif data == 'action_tts':
        text = (
            "🔊 **Text to Speech**\n\n"
            "📝 Please send the text you want to convert to speech.\n\n"
            "💡 **Example:**\n"
            "Just type: `Hello, this is a test`\n\n"
            "Or use the command:\n"
            "`/tts Your text here`"
        )
        keyboard = get_text_tools_keyboard()
        context.user_data['awaiting_tts'] = True
    
    elif data == 'action_translate':
        text = (
            "🌍 **Translator**\n\n"
            "📝 Use the command:\n"
            "`/translate [language] [text]`\n\n"
            "💡 **Examples:**\n"
            "`/translate es Hello World` (to Spanish)\n"
            "`/translate fr Good morning` (to French)\n"
            "`/translate de Thank you` (to German)\n"
            "`/translate ar Hello` (to Arabic)\n\n"
            "🌐 Supports 100+ languages!"
        )
        keyboard = get_text_tools_keyboard()
    
    elif data == 'action_encrypt':
        text = (
            "🔐 **Encrypt Text**\n\n"
            "📝 Use the command:\n"
            "`/encrypt [password] [text]`\n\n"
            "💡 **Example:**\n"
            "`/encrypt mypass123 Secret message`\n\n"
            "🔒 Your text will be encrypted with military-grade security!"
        )
        keyboard = get_text_tools_keyboard()
    
    elif data == 'action_decrypt':
        text = (
            "🔓 **Decrypt Text**\n\n"
            "📝 Use the command:\n"
            "`/decrypt [password] [encrypted text]`\n\n"
            "💡 **Remember:**\n"
            "Use the **same password** you used for encryption!\n\n"
            "🔑 Paste the encrypted text from the encrypt command."
        )
        keyboard = get_text_tools_keyboard()
    
    elif data == 'action_ytdl':
        # Set user state to expect YouTube URL
        context.user_data['awaiting_ytdl_url'] = True
        text = (
            "📺 **Advanced Video Downloader**\n\n"
            "📝 **Please send me a video URL:**\n\n"
            "🌐 **Supported Platforms:**\n"
            "• YouTube, TikTok, Instagram\n"
            "• Facebook, Twitter/X, Vimeo\n"
            "• And 1000+ more sites!\n\n"
            "💡 **Examples:**\n"
            "`https://youtu.be/xxxxx`\n"
            "`https://www.youtube.com/watch?v=xxxxx`\n"
            "`https://www.tiktok.com/@user/video/xxxxx`\n"
            "`https://www.instagram.com/reel/xxxxx`\n\n"
            "✨ **After sending URL, choose:**\n"
            "🎵 Audio (MP3/Best Quality)\n"
            "📺 Video (360p/480p/720p/Best)\n\n"
            "📦 **Maximum File Size:** 2GB\n\n"
            "⬅️ Send the URL now, or click Back to cancel"
        )
        keyboard = get_video_downloader_back_only()  # Only show Back button
    
    elif data == 'action_igdl':
        # Set user state to expect Instagram URL
        context.user_data['awaiting_igdl_url'] = True
        text = (
            "📸 **Instagram Downloader**\n\n"
            "📝 **Please send me an Instagram URL:**\n\n"
            "💡 **Examples:**\n"
            "`https://www.instagram.com/p/xxxxx`\n"
            "`https://www.instagram.com/reel/xxxxx`\n\n"
            "✨ Downloads:\n"
            "📷 Photos\n"
            "🎬 Videos\n"
            "🎞️ Reels\n\n"
            "📦 **Download Limit:** Up to 1GB\n\n"
            "⬅️ Send the URL now, or click Back to cancel"
        )
        keyboard = get_downloader_keyboard()
    
    elif data == 'action_fbdl':
        # Set user state to expect Facebook URL
        context.user_data['awaiting_fbdl_url'] = True
        text = (
            "📘 **Facebook Downloader**\n\n"
            "📝 **Please send me a Facebook video URL:**\n\n"
            "💡 **Examples:**\n"
            "`https://www.facebook.com/watch/?v=xxxxx`\n"
            "`https://fb.watch/xxxxx`\n\n"
            "✨ Downloads Facebook videos in best quality\n\n"
            "📦 **Download Limit:** Up to 1GB\n\n"
            "⬅️ Send the URL now, or click Back to cancel"
        )
        keyboard = get_downloader_keyboard()
    
    elif data == 'action_shorten':
        text = (
            "🔗 **URL Shortener**\n\n"
            "📝 Use the command:\n"
            "`/shorten [URL]`\n\n"
            "💡 **Example:**\n"
            "`/shorten https://example.com/very/long/url`\n\n"
            "⚡ Get an instant short link!"
        )
        keyboard = get_downloader_keyboard()
    
    elif data == 'action_qr':
        text = (
            "📱 **QR Code Generator**\n\n"
            "📝 Use the command:\n"
            "`/qr [text or URL]`\n\n"
            "💡 **Examples:**\n"
            "`/qr https://example.com`\n"
            "`/qr Contact: +1234567890`\n"
            "`/qr My WiFi Password`\n\n"
            "📷 Generates a high-quality QR code image!"
        )
        keyboard = get_downloader_keyboard()

    elif data.startswith('delete_stored_'):
        # Inline deletion of stored file
        user_id = query.from_user.id
        try:
            fid = int(data.split('_')[-1])
        except ValueError:
            text = "❌ Invalid delete request."
            keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='menu_storage')]])
        else:
            # Fetch file info
            result = db_pool.execute(
                "SELECT filename, drive_file_id FROM stored_files WHERE id = ? AND user_id = ?",
                (fid, user_id)
            )
            if not result:
                text = (
                    "❌ **File Not Found**\n\n"
                    f"🆔 ID `{fid}` doesn't exist or already deleted."
                )
                keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("📂 My Files", callback_data='action_list_files'), InlineKeyboardButton("⬅️ Back", callback_data='menu_storage')]])
            else:
                filename, drive_file_id = result[0]
                # Try Drive deletion
                drive_service = get_drive_service()
                if drive_service and drive_file_id:
                    try:
                        drive_service.files().delete(fileId=drive_file_id).execute()
                    except Exception as de:
                        logger.warning(f"Drive delete failed for {drive_file_id}: {de}")
                # Remove from DB
                db_pool.execute(
                    "DELETE FROM stored_files WHERE id = ? AND user_id = ?",
                    (fid, user_id),
                    fetch='none'
                )
                text = (
                    "🗑️ **File Deleted Successfully!**\n\n"
                    f"📄 **Filename:** `{filename}`\n"
                    "☁️ Removed from Google Drive\n"
                    "🗃️ Removed from storage list\n\n"
                    "📂 Use 'My Files' to view remaining files."
                )
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("📂 My Files", callback_data='action_list_files'), InlineKeyboardButton("📤 Upload Another", callback_data='action_upload_file')],
                    [InlineKeyboardButton("⬅️ Back", callback_data='menu_storage')]
                ])
    
    elif data.startswith('compress_'):
        # Handle compression quality selection
        user_id = query.from_user.id
        quality_name = data.replace('compress_', '')
        
        # Set quality based on selection
        quality_map = {
            'high': (85, 'High'),
            'medium': (70, 'Medium'),
            'low': (50, 'Low')
        }
        
        quality, quality_display = quality_map.get(quality_name, (85, 'High'))
        
        # Set state and quality
        session_manager.set_state(user_id, 'awaiting_compress_image')
        context.user_data['compress_quality'] = quality
        context.user_data['compress_quality_name'] = quality_name
        
        text = (
            f"🗜️ **Image Compressor**\n\n"
            f"✅ **Quality Selected:** {quality_display} ({quality}%)\n\n"
            f"📸 **Now send me a photo to compress!**\n\n"
            f"💡 The file size will be reduced while maintaining quality."
        )
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Change Quality", callback_data='help_compress')],
            [InlineKeyboardButton("⬅️ Back", callback_data='menu_file')]
        ])
    
    elif data == 'done_sending_photos':
        # User finished sending all photos, show Create PDF button
        user_id = query.from_user.id
        logger.info(f"DONE_SENDING button clicked by user {user_id}")
        
        # Check if user is in photo collection mode
        if session_manager.get_state(user_id) != 'collecting_photos_for_pdf':
            await query.answer(
                "❌ You're not collecting photos. Use Photos to PDF feature first.",
                show_alert=True
            )
            return
        
        photos = context.user_data.get('pdf_photos', [])
        
        if not photos:
            await query.answer(
                "❌ No photos collected yet! Send at least one photo first.",
                show_alert=True
            )
            return
        
        # Show Create PDF button
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Create PDF Now", callback_data='create_pdf')],
            [InlineKeyboardButton("❌ Cancel", callback_data='menu_file')]
        ])
        
        await query.answer()
        # Send new message instead of editing (to avoid "Message to edit not found" error)
        try:
            await query.edit_message_text(
                f"✅ **All photos captured!**\n\n"
                f"📸 Total photos: {len(photos)}\n\n"
                f"📄 Click 'Create PDF' to generate your PDF file.",
                reply_markup=keyboard,
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception:
            # If edit fails, send a new message
            await query.message.reply_text(
                f"✅ **All photos captured!**\n\n"
                f"📸 Total photos: {len(photos)}\n\n"
                f"📄 Click 'Create PDF' to generate your PDF file.",
                reply_markup=keyboard,
                parse_mode=ParseMode.MARKDOWN
            )
        return
    
    elif data == 'create_pdf':
        # Handle PDF creation from collected photos
        user_id = query.from_user.id
        logger.info(f"CREATE_PDF button clicked by user {user_id}")
        
        # Check if user is in photo collection mode
        if session_manager.get_state(user_id) != 'collecting_photos_for_pdf':
            logger.warning(f"User {user_id} not in photo collection mode. Current state: {session_manager.get_state(user_id)}")
            await query.answer(
                "❌ You're not collecting photos. Use Photos to PDF feature first.",
                show_alert=True
            )
            return
        
        photos = context.user_data.get('pdf_photos', [])
        logger.info(f"User {user_id} has {len(photos)} photos collected")
        
        if not photos:
            logger.warning(f"User {user_id} has no photos collected")
            await query.answer(
                "❌ No photos collected yet! Send at least one photo first.",
                show_alert=True
            )
            return
        
        # Clear state
        session_manager.clear_state(user_id)
        context.user_data['pdf_photos'] = []
        
        # Check if user is busy
        if task_lock.is_user_busy(user_id):
            active_task = task_lock.get_active_task(user_id)
            await query.answer(
                f"⏳ Please wait! Active task: {active_task}",
                show_alert=True
            )
            return
        
        if not task_lock.lock_user(user_id, "PDF Creation"):
            await query.answer("⚠️ Unable to start task. Please try again.", show_alert=True)
            return
        
        # Edit the message to show loading
        await query.edit_message_text(
            f"📄 Creating PDF with {len(photos)} photo(s)...\n\n"
            "⏳ Please wait...",
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Continue with PDF creation - call the existing done_command logic
        # We'll create the PDF here inline
        try:
            timestamp = int(time.time())
            pdf_path = f"photos_{user_id}_{timestamp}.pdf"
            
            # Download all photos
            await query.edit_message_text(
                f"📥 Downloading {len(photos)} photo(s)...",
                parse_mode=ParseMode.MARKDOWN
            )
            
            photo_files = []
            for idx, photo_id in enumerate(photos, 1):
                file = await context.bot.get_file(photo_id)
                photo_data = io.BytesIO()
                await file.download_to_memory(photo_data)
                photo_data.seek(0)
                photo_files.append(photo_data)
                
                if idx % 5 == 0:
                    await query.edit_message_text(
                        f"📥 Downloaded {idx}/{len(photos)} photos...",
                        parse_mode=ParseMode.MARKDOWN
                    )
            
            # Create PDF
            await query.edit_message_text(
                "📄 Creating PDF document...",
                parse_mode=ParseMode.MARKDOWN
            )
            
            loop = asyncio.get_event_loop()
            
            def create_pdf_from_images(photo_data_list, output_path):
                """Create PDF from images"""
                c = canvas.Canvas(output_path, pagesize=A4)
                page_width, page_height = A4
                
                for photo_data in photo_data_list:
                    photo_data.seek(0)
                    img = Image.open(photo_data)
                    
                    # Calculate scaling to fit A4
                    img_width, img_height = img.size
                    aspect = img_height / img_width
                    
                    # Fit to page with margin
                    margin = 20
                    max_width = page_width - (2 * margin)
                    max_height = page_height - (2 * margin)
                    
                    if img_width > max_width or img_height > max_height:
                        if aspect > 1:  # Portrait
                            new_height = min(max_height, img_height)
                            new_width = new_height / aspect
                        else:  # Landscape
                            new_width = min(max_width, img_width)
                            new_height = new_width * aspect
                    else:
                        new_width = img_width
                        new_height = img_height
                    
                    # Center image
                    x = (page_width - new_width) / 2
                    y = (page_height - new_height) / 2
                    
                    # Save image temporarily
                    temp_img = io.BytesIO()
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    img.save(temp_img, format='JPEG', quality=95)
                    temp_img.seek(0)
                    
                    c.drawImage(ImageReader(temp_img), x, y, width=new_width, height=new_height)
                    c.showPage()
                
                c.save()
            
            await loop.run_in_executor(executor, create_pdf_from_images, photo_files, pdf_path)
            
            # Send PDF
            await query.edit_message_text(
                "📤 Uploading PDF...",
                parse_mode=ParseMode.MARKDOWN
            )
            
            with open(pdf_path, 'rb') as pdf_file:
                await context.bot.send_document(
                    chat_id=query.message.chat_id,
                    document=pdf_file,
                    caption=f"✅ PDF created with {len(photos)} photo(s)!",
                    filename=f"photos_{timestamp}.pdf"
                )
            
            # Success message
            await query.edit_message_text(
                "✅ **PDF Created Successfully!**\n\n"
                f"📄 {len(photos)} photos combined\n\n"
                "🎯 Returning to main menu...",
                reply_markup=get_main_menu_keyboard(),
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Clean up
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            
        except Exception as e:
            logger.error(f"Error creating PDF: {e}")
            await query.edit_message_text(
                "❌ Failed to create PDF. Please try again.",
                reply_markup=get_main_menu_keyboard(),
                parse_mode=ParseMode.MARKDOWN
            )
        finally:
            task_lock.unlock_user(user_id)
        
        return
    
    # Try to edit the message - handle both text and photo messages
    try:
        if query.message.photo:
            # This is a photo message - delete it and send new text message
            logger.info(f"Deleting photo message and sending text for: {data}")
            await query.message.delete()
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
        else:
            # Regular text message - just edit it
            await query.edit_message_text(text=text, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
    except Exception as e:
        logger.error(f"Error editing message in button_callback_handler: {e}")
        # Fallback: try to send a new message
        try:
            await query.message.delete()
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            await query.answer("❌ Error. Please use /start to restart.", show_alert=True)

# --- All other feature functions (handle_photo, tts_command, etc.) are unchanged ---
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    logger.info(f"PHOTO RECEIVED from user {user_id}. Current state: {session_manager.get_state(user_id)}")
    
    # If user is in File Storage upload flow, route photo to storage handler
    if session_manager.get_state(user_id) == 'awaiting_file_upload':
        # Delegate to file upload handler and stop further processing
        await handle_file_upload(update, context)
        return
    
    # Check if user is submitting payment screenshot
    if session_manager.get_state(user_id) == 'awaiting_payment_screenshot':
        file_id = update.message.photo[-1].file_id
        session = session_manager.get_session(user_id)
        session['data']['screenshot_file_id'] = file_id
        session_manager.set_state(user_id, 'awaiting_payment_reference')
        
        await update.message.reply_text(
            "✅ *Screenshot Received!*\n\n"
            "Now please send your *UPI Reference Number*\n"
            "(Transaction ID from your payment app)\n\n"
            "💡 Example: `123456789012`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Check if admin is setting QR code
    if user_id == ADMIN_ID and session_manager.get_state(user_id) == 'awaiting_qr_code':
        file_id = update.message.photo[-1].file_id
        set_setting('qr_code_file_id', file_id)
        session_manager.clear_state(user_id)
        
        await update.message.reply_text(
            "✅ *QR Code Updated!*\n\n"
            "The new QR code will be shown to users requesting premium.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Check if user is collecting photos for PDF
    if session_manager.get_state(user_id) == 'collecting_photos_for_pdf':
        file_id = update.message.photo[-1].file_id
        photos = context.user_data.get('pdf_photos', [])
        photos.append(file_id)
        context.user_data['pdf_photos'] = photos
        
        logger.info(f"User {user_id} added photo to PDF collection. Total: {len(photos)} photos")
        
        # Show "Done Sending" button only after the first photo, with minimal text
        if len(photos) == 1:
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Done Sending", callback_data='done_sending_photos')],
                [InlineKeyboardButton("❌ Cancel", callback_data='menu_file')]
            ])
            
            await update.message.reply_text(
                "📸 Send more photos or click 'Done Sending'",
                reply_markup=keyboard
            )
        # For subsequent photos, collect silently (no message, no button)
        return
    
    # Check if user wants to compress image
    if session_manager.get_state(user_id) == 'awaiting_compress_image':
        file_id = update.message.photo[-1].file_id
        quality = context.user_data.get('compress_quality', 85)
        quality_name = context.user_data.get('compress_quality_name', 'high')
        
        # Clear state
        session_manager.clear_state(user_id)
        
        # Check if user is busy
        if task_lock.is_user_busy(user_id):
            active_task = task_lock.get_active_task(user_id)
            await update.message.reply_text(
                f"⏳ Please wait! You have an active task: *{active_task}*",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        if not task_lock.lock_user(user_id, "Image Compression"):
            await update.message.reply_text("⚠️ Unable to start task. Please try again.")
            return
        
        loading_msg = await update.message.reply_text("🗜️ Compressing image...")
        
        try:
            file = await context.bot.get_file(file_id)
            
            # Download to memory
            image_data = io.BytesIO()
            await file.download_to_memory(image_data)
            image_data.seek(0)
            
            await loading_msg.edit_text(f"🎨 Compressing ({quality_name} quality)...")
            
            # Compress in thread pool
            loop = asyncio.get_event_loop()
            
            def compress_image(img_data, quality_val):
                """Compress image"""
                img_data.seek(0)
                img = Image.open(img_data)
                
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                
                # Compress
                compressed = io.BytesIO()
                img.save(compressed, format='JPEG', quality=quality_val, optimize=True)
                compressed.seek(0)
                
                return compressed, len(img_data.getvalue()), len(compressed.getvalue())
            
            compressed_data, original_size, compressed_size = await loop.run_in_executor(
                executor, compress_image, image_data, quality
            )
            
            await loading_msg.edit_text("📤 Uploading compressed image...")
            
            # Calculate compression ratio
            ratio = ((original_size - compressed_size) / original_size) * 100
            
            # Send compressed image
            await update.message.reply_photo(
                photo=compressed_data,
                caption=(
                    f"✅ **Image Compressed!**\n\n"
                    f"📊 **Quality:** {quality_name.capitalize()} ({quality}%)\n"
                    f"📉 **Original:** {original_size / 1024:.1f} KB\n"
                    f"📈 **Compressed:** {compressed_size / 1024:.1f} KB\n"
                    f"💾 **Saved:** {ratio:.1f}% smaller"
                ),
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Show success with main menu
            await loading_msg.edit_text(
                "✅ *Compression complete!*\n\n"
                "🎯 Returning to main menu...",
                reply_markup=get_main_menu_keyboard(),
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Auto-return to main menu
            await asyncio.sleep(1.5)
            await loading_msg.edit_text(
                "🤖 *Welcome to Super Bot!*\n\n"
                "Choose a feature from the menu below:",
                reply_markup=get_main_menu_keyboard(),
                parse_mode=ParseMode.MARKDOWN
            )
            
        except Exception as e:
            logger.error(f"Error in image compression: {e}")
            await loading_msg.edit_text(
                f"❌ Compression failed: {str(e)}",
                reply_markup=get_main_menu_keyboard()
            )
        finally:
            task_lock.unlock_user(user_id)
        
        return
    
    # Check if user is verified
    if not is_user_verified(user_id):
        await update.message.reply_text(
            "🔒 **Verification Required**\n\n"
            "Please use /start to verify your account first.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Regular image conversion
    file_id = update.message.photo[-1].file_id
    
    # Check if user is busy with another task
    if task_lock.is_user_busy(user_id):
        active_task = task_lock.get_active_task(user_id)
        await update.message.reply_text(
            f"⏳ *Please wait!*\n\nYou have an active task: *{active_task}*\n\n"
            "Please wait for it to complete before starting a new one.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Check usage limit
    can_use, msg = can_use_tool(user_id)
    if not can_use:
        keyboard = [[InlineKeyboardButton("💎 Get Premium", callback_data='get_premium')]]
        await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
        return
    
    # Lock user for image conversion task
    if not task_lock.lock_user(user_id, "Image Conversion"):
        await update.message.reply_text("⏳ Please wait for your previous task to complete!")
        return
    
    try:
        # Smooth animation for image processing
        loading_msg = await update.message.reply_text("🖼️ Preparing image...")
        await asyncio.sleep(0.5)
        
        file = await context.bot.get_file(file_id)
        await loading_msg.edit_text("📥 Downloading image...")
        await asyncio.sleep(0.3)
        
        # Download to memory
        image_data = io.BytesIO()
        await file.download_to_memory(image_data)
        image_data.seek(0)
        
        # Process image in thread pool (non-blocking)
        await loading_msg.edit_text("🎨 Converting to PNG...")
        await asyncio.sleep(0.3)
        loop = asyncio.get_event_loop()
        
        def process_image(img_data):
            """CPU-intensive image processing"""
            img_data.seek(0)
            img = Image.open(img_data)
            
            # PNG conversion
            png_buffer = io.BytesIO()
            img.save(png_buffer, format='PNG')
            png_buffer.seek(0)
            
            # WEBP conversion
            webp_buffer = io.BytesIO()
            img.save(webp_buffer, format='WEBP')
            webp_buffer.seek(0)
            
            return png_buffer, webp_buffer
        
        # Run in thread pool for true parallel processing
        png_buffer, webp_buffer = await loop.run_in_executor(executor, process_image, image_data)
        
        await loading_msg.edit_text("🌐 Converting to WEBP...")
        await asyncio.sleep(0.3)
        
        await loading_msg.edit_text("📤 Uploading converted images...")
        await update.message.reply_document(document=png_buffer, filename=f'image_{file_id}.png', caption="✅ PNG format")
        await update.message.reply_document(document=webp_buffer, filename=f'image_{file_id}.webp', caption="✅ WEBP format")
        
        # Show success with main menu
        await loading_msg.edit_text(
            "✅ *Conversion complete!* Both formats sent.\n\n"
            "🎯 Returning to main menu...",
            reply_markup=get_main_menu_keyboard(),
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Auto-return to main menu after 1.5 seconds
        await asyncio.sleep(1.5)
        await loading_msg.edit_text(
            "🤖 *Welcome to Super Bot!*\n\n"
            "Choose a feature from the menu below:",
            reply_markup=get_main_menu_keyboard(),
            parse_mode=ParseMode.MARKDOWN
        )
        
        increment_usage(user_id)  # Count this usage
    except Exception as e:
        logger.error(f"Error in handle_photo: {e}")
        await loading_msg.edit_text(
            "❌ Could not process image. Please try again.",
            reply_markup=get_main_menu_keyboard()
        )
    finally:
        # Unlock user after task completion
        task_lock.unlock_user(user_id)

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    # If user is in File Storage upload flow, route video to storage handler
    if session_manager.get_state(user_id) == 'awaiting_file_upload':
        await handle_file_upload(update, context)
        return
    
    # Check if user is verified
    if not is_user_verified(user_id):
        await update.message.reply_text(
            "🔒 **Verification Required**\n\n"
            "Please use /start to verify your account first.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    timestamp = int(time.time())
    file_id = update.message.video.file_id
    # Unique filenames per user to prevent conflicts
    file_path = f"{user_id}_{timestamp}_{file_id}.mp4"
    audio_path = f"{user_id}_{timestamp}_{file_id}.mp3"
    
    # Check if user is busy with another task
    if task_lock.is_user_busy(user_id):
        active_task = task_lock.get_active_task(user_id)
        await update.message.reply_text(
            f"⏳ *Please wait!*\n\nYou have an active task: *{active_task}*\n\n"
            "Please wait for it to complete before starting a new one.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Lock user for video conversion task
    if not task_lock.lock_user(user_id, "Video to MP3"):
        await update.message.reply_text("⏳ Please wait for your previous task to complete!")
        return
    
    try:
        # Smooth animation for video processing
        loading_msg = await update.message.reply_text("🎬 Preparing video...")
        await asyncio.sleep(0.5)
        
        file = await context.bot.get_file(file_id)
        await loading_msg.edit_text("📥 Downloading video...")
        await asyncio.sleep(0.3)
        await file.download_to_drive(file_path)
        
        await loading_msg.edit_text("🎵 Extracting audio track...")
        await asyncio.sleep(0.3)
        
        # Extract audio in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        
        def extract_audio(video_path, audio_path):
            """CPU-intensive video processing"""
            video_clip = VideoFileClip(video_path)
            video_clip.audio.write_audiofile(audio_path, logger=None)
            video_clip.close()
        
        await loop.run_in_executor(executor, extract_audio, file_path, audio_path)
        
        await loading_msg.edit_text("🎶 Converting to MP3...")
        await asyncio.sleep(0.3)
        
        await loading_msg.edit_text("📤 Uploading audio file...")
        with open(audio_path, 'rb') as audio_file:
            await update.message.reply_audio(audio=audio_file, title="Converted Audio", caption="✅ Audio extracted successfully!")
        
        # Show success with main menu
        await loading_msg.edit_text(
            "✅ *Video to MP3 conversion complete!*\n\n"
            "🎯 Returning to main menu...",
            reply_markup=get_main_menu_keyboard(),
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Auto-return to main menu after 1.5 seconds
        await asyncio.sleep(1.5)
        await loading_msg.edit_text(
            "🤖 *Welcome to Super Bot!*\n\n"
            "Choose a feature from the menu below:",
            reply_markup=get_main_menu_keyboard(),
            parse_mode=ParseMode.MARKDOWN
        )
        
    except Exception as e:
        logger.error(f"Error in handle_video: {e}")
        await loading_msg.edit_text(
            "❌ Failed to convert video. File might be too large or corrupted.",
            reply_markup=get_main_menu_keyboard()
        )
    finally:
        # Unlock user after task completion
        task_lock.unlock_user(user_id)
        if os.path.exists(file_path): os.remove(file_path)
        if os.path.exists(audio_path): os.remove(audio_path)

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    # If user is in File Storage upload flow, treat incoming PDF as a storage upload
    if session_manager.get_state(user_id) == 'awaiting_file_upload':
        await handle_file_upload(update, context)
        return
    
    # Check if user is verified
    if not is_user_verified(user_id):
        await update.message.reply_text(
            "🔒 **Verification Required**\n\n"
            "Please use /start to verify your account first.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Check if user is busy with another task
    if task_lock.is_user_busy(user_id):
        active_task = task_lock.get_active_task(user_id)
        await update.message.reply_text(
            f"⏳ Please wait! You have an active task: *{active_task}*\n\n"
            "Try again once your current task completes.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Lock user for this task
    if not task_lock.lock_user(user_id, "PDF Conversion"):
        await update.message.reply_text("⚠️ Unable to start task. Please try again.")
        return
    
    timestamp = int(time.time())
    file_id = update.message.document.file_id
    # Unique filenames per user to prevent conflicts
    pdf_path = f"{user_id}_{timestamp}_{file_id}.pdf"
    docx_path = f"{user_id}_{timestamp}_{file_id}.docx"
    
    # Smooth animation for PDF processing
    loading_msg = await update.message.reply_text("📄 Preparing PDF...")
    await asyncio.sleep(0.5)
    
    try:
        file = await context.bot.get_file(file_id)
        await loading_msg.edit_text("📥 Downloading PDF...")
        await asyncio.sleep(0.3)
        await file.download_to_drive(pdf_path)
        
        await loading_msg.edit_text("� Reading PDF pages...")
        await asyncio.sleep(0.3)
        
        await loading_msg.edit_text("✍️ Converting to DOCX format...")
        
        # Convert PDF in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        
        def convert_pdf(pdf_path, docx_path):
            """CPU-intensive PDF processing"""
            doc = Document()
            with fitz.open(pdf_path) as pdf_doc:
                for page in pdf_doc:
                    doc.add_paragraph(page.get_text())
            doc.save(docx_path)
        
        await loop.run_in_executor(executor, convert_pdf, pdf_path, docx_path)
        
        await loading_msg.edit_text("📝 Finalizing document...")
        await asyncio.sleep(0.3)
        
        await loading_msg.edit_text("📤 Uploading converted document...")
        
        with open(docx_path, 'rb') as docx_file:
            await update.message.reply_document(document=docx_file, caption="✅ PDF converted to DOCX successfully!")
        
        # Show success with main menu
        await loading_msg.edit_text(
            "✅ *PDF conversion complete!*\n\n"
            "🎯 Returning to main menu...",
            reply_markup=get_main_menu_keyboard(),
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Auto-return to main menu after 1.5 seconds
        await asyncio.sleep(1.5)
        await loading_msg.edit_text(
            "🤖 *Welcome to Super Bot!*\n\n"
            "Choose a feature from the menu below:",
            reply_markup=get_main_menu_keyboard(),
            parse_mode=ParseMode.MARKDOWN
        )
        
    except Exception as e:
        logger.error(f"Error in handle_pdf: {e}")
        await loading_msg.edit_text(
            "❌ Failed to convert PDF. File might be too large or corrupted.",
            reply_markup=get_main_menu_keyboard()
        )
    finally:
        # Always unlock the user
        task_lock.unlock_user(user_id)
        if os.path.exists(pdf_path): os.remove(pdf_path)
        if os.path.exists(docx_path): os.remove(docx_path)

async def handle_file_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle file uploads for storage"""
    user_id = update.effective_user.id
    
    # Check if user is verified
    if not is_user_verified(user_id):
        await update.message.reply_text(
            "🔒 **Verification Required**\n\n"
            "Please use /start to verify your account first.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Check if user is in file upload state
    if not session_manager.get_state(user_id) == 'awaiting_file_upload':
        return  # Not awaiting file upload, ignore
    
    # Clear the state
    session_manager.clear_state(user_id)
    context.user_data['awaiting_file_upload'] = False
    
    # Check if user is busy with another task
    if task_lock.is_user_busy(user_id):
        active_task = task_lock.get_active_task(user_id)
        await update.message.reply_text(
            f"⏳ Please wait! You have an active task: *{active_task}*\n\n"
            "Try again once your current task completes.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Lock user for this task
    if not task_lock.lock_user(user_id, "File Upload"):
        await update.message.reply_text("⚠️ Unable to start upload. Please try again.")
        return
    
    loading_msg = await update.message.reply_text("📤 Uploading file to storage...")
    
    try:
        # Get file info
        if update.message.document:
            file = update.message.document
            file_obj = await context.bot.get_file(file.file_id)
            filename = file.file_name
            file_size = file.file_size
        elif update.message.photo:
            file = update.message.photo[-1]  # Get largest photo
            file_obj = await context.bot.get_file(file.file_id)
            filename = f"photo_{int(time.time())}.jpg"
            file_size = file.file_size
        elif update.message.video:
            file = update.message.video
            file_obj = await context.bot.get_file(file.file_id)
            filename = file.file_name or f"video_{int(time.time())}.mp4"
            file_size = file.file_size
        elif update.message.audio:
            file = update.message.audio
            file_obj = await context.bot.get_file(file.file_id)
            filename = file.file_name or f"audio_{int(time.time())}.mp3"
            file_size = file.file_size
        else:
            await loading_msg.edit_text("❌ Unsupported file type. Please send a document, photo, video, or audio file.")
            task_lock.unlock_user(user_id)
            return
        
        # Download file
        timestamp = int(time.time())
        local_path = f"{user_id}_{timestamp}_{filename}"
        
        await loading_msg.edit_text("📥 Downloading file...")
        await file_obj.download_to_drive(local_path)
        
        # Upload to Google Drive
        await loading_msg.edit_text("☁️ Uploading to Google Drive...")
        
        # Prepare Google Drive service BEFORE try so NameError cannot occur
        drive_service = get_drive_service()
        if not drive_service:
            await loading_msg.edit_text(
                "❌ **Google Drive Authentication Failed**\n\n"
                "• Missing or invalid credentials.json / token.json\n"
                "• Or OAuth consent not completed.\n\n"
                "Fix: Place credentials.json and run a simple upload to generate token.json.",
                parse_mode=ParseMode.MARKDOWN
            )
            task_lock.unlock_user(user_id)
            return

        try:
            # Upload file to Google Drive
            file_metadata = {
                'name': filename,
                'parents': [DRIVE_FOLDER_ID] if DRIVE_FOLDER_ID else []
            }
            
            media = MediaFileUpload(local_path, resumable=True)
            drive_file = drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, webViewLink'
            ).execute()
            
            drive_file_id = drive_file.get('id')
            
            # Make file publicly accessible
            drive_service.permissions().create(
                fileId=drive_file_id,
                body={'type': 'anyone', 'role': 'reader'}
            ).execute()
            
            # Get shareable link
            share_link = f"https://drive.google.com/file/d/{drive_file_id}/view?usp=sharing"
            
            # Save to database and capture inserted row id
            inserted_id = db_pool.execute(
                "INSERT INTO stored_files (user_id, filename, file_size, drive_file_id, share_link) VALUES (?, ?, ?, ?, ?)",
                (user_id, filename, file_size, drive_file_id, share_link),
                fetch='insert'
            )
            
            # Format file size
            size_mb = file_size / (1024 * 1024)
            if size_mb < 1:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{size_mb:.1f} MB"
            
            success_text = (
                f"✅ **File Uploaded Successfully!**\n\n"
                f"📄 **Name:** `{filename}`\n"
                f"📊 **Size:** {size_str}\n"
                f"☁️ **Storage:** Google Drive\n\n"
                f"🔗 **Share Link:**\n"
                f"{share_link}\n\n"
                f"💡 Access this file anytime from 'My Files'!"
            )
            
            # Provide quick action buttons including Delete
            success_keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🗑️ Delete This File", callback_data=f"delete_stored_{inserted_id}")],
                [
                    InlineKeyboardButton("📂 My Files", callback_data='action_list_files'),
                    InlineKeyboardButton("⬅️ Back", callback_data='menu_storage')
                ]
            ])

            await loading_msg.edit_text(
                success_text,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
                reply_markup=success_keyboard
            )
            
        except Exception as drive_error:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Google Drive upload error: {drive_error}\nTRACEBACK:\n{tb}")
            await loading_msg.edit_text(
                "❌ **Failed to upload to Google Drive**\n\n"
                "Please check Google Drive configuration or try again later.",
                parse_mode=ParseMode.MARKDOWN
            )
        finally:
            # Clean up local file with retry logic
            if os.path.exists(local_path):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        await asyncio.sleep(0.5)  # Small delay to ensure file handle is released
                        os.remove(local_path)
                        break
                    except PermissionError:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)  # Wait longer before retry
                        else:
                            logger.warning(f"Could not delete {local_path} - file locked. Will be cleaned up later.")
                    except Exception as del_error:
                        logger.error(f"Error deleting file {local_path}: {del_error}")
                        break
        
    except Exception as e:
        logger.error(f"Error in handle_file_upload: {e}")
        try:
            await loading_msg.edit_text(f"❌ Upload failed: {str(e)}")
        except:
            pass
        # Try to clean up local file even on error
        if 'local_path' in locals() and os.path.exists(local_path):
            try:
                await asyncio.sleep(0.5)
                os.remove(local_path)
            except:
                logger.warning(f"Could not delete {local_path} on error - will be cleaned up later.")
    finally:
        task_lock.unlock_user(user_id)

async def tts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    # Check if user is verified
    if not is_user_verified(user_id):
        await update.message.reply_text(
            "🔒 **Verification Required**\n\n"
            "Please use /start to verify your account first.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    text = " ".join(context.args)
    if not text:
        await update.message.reply_text("❌ **Usage:** `/tts [text]`\n\n💡 **Example:** `/tts Hello World`", parse_mode=ParseMode.MARKDOWN)
        return
    
    # Check if user is busy
    if task_lock.is_user_busy(user_id):
        active_task = task_lock.get_active_task(user_id)
        await update.message.reply_text(
            f"⏳ Please wait! You have an active task: *{active_task}*",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    if not task_lock.lock_user(user_id, "Text to Speech"):
        return
    
    loading_msg = await update.message.reply_text("🔊 Generating voice...")
    try:
        tts = gTTS(text=text, lang='en')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        await loading_msg.edit_text("📤 Sending voice message...")
        await update.message.reply_voice(voice=audio_buffer, caption="✅ Text to Speech")
        await loading_msg.delete()
    except Exception as e:
        await loading_msg.edit_text("❌ Failed to generate voice. Please try again.")
    finally:
        task_lock.unlock_user(user_id)

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ **Usage:** `/translate <lang> <text>`\n\n"
                "💡 **Examples:**\n"
                "`/translate es Hello World` (to Spanish)\n"
                "`/translate fr Good morning` (to French)\n"
                "`/translate de Thank you` (to German)\n\n"
                "🌍 **Popular codes:** en, es, fr, de, it, pt, ru, zh, ja, ko",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        # Check if user is busy
        if task_lock.is_user_busy(user_id):
            active_task = task_lock.get_active_task(user_id)
            await update.message.reply_text(
                f"⏳ Please wait! You have an active task: *{active_task}*",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        if not task_lock.lock_user(user_id, "Translation"):
            return
        
        lang_code, text = context.args[0], " ".join(context.args[1:])
        loading_msg = await update.message.reply_text("🌍 Translating...")
        
        try:
            translated = GoogleTranslator(source='auto', target=lang_code).translate(text)
            await loading_msg.edit_text(
                f"✅ **Translated to '{lang_code.upper()}':**\n\n{translated}",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as trans_error:
            logger.error(f"Translation error: {trans_error}")
            await loading_msg.edit_text(
                f"❌ Translation failed.\n\n"
                f"**Possible reasons:**\n"
                f"• Invalid language code '{lang_code}'\n"
                f"• Connection issue\n"
                f"• Text too long\n\n"
                f"💡 Try: /translate en {text[:20]}..."
            )
    except Exception as e:
        logger.error(f"Translate command error: {e}")
        await update.message.reply_text("❌ Translation failed. Check the language code and try again.")
    finally:
        task_lock.unlock_user(user_id)

def get_key_from_password(password: str):
    salt = b'fixed_salt'
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

async def encrypt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if len(context.args) < 2:
        await update.message.reply_text(
            "❌ **Usage:** `/encrypt <password> <text>`\n\n"
            "💡 **Example:** `/encrypt mypass Secret message`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Check if user is busy
    if task_lock.is_user_busy(user_id):
        active_task = task_lock.get_active_task(user_id)
        await update.message.reply_text(
            f"⏳ Please wait! You have an active task: *{active_task}*",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    if not task_lock.lock_user(user_id, "Encryption"):
        return
    
    password, text = context.args[0], " ".join(context.args[1:])
    loading_msg = await update.message.reply_text("🔐 Encrypting...")
    
    try:
        fernet = Fernet(get_key_from_password(password))
        encrypted = fernet.encrypt(text.encode())
        await loading_msg.edit_text(
            f"✅ **Encrypted Text:**\n\n`{encrypted.decode()}`\n\n"
            f"💡 Use the same password to decrypt!",
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        await loading_msg.edit_text("❌ Encryption failed. Please try again.")
    finally:
        task_lock.unlock_user(user_id)

async def decrypt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ **Usage:** `/decrypt <password> <encrypted_text>`\n\n"
                "💡 Use the same password used for encryption",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        # Check if user is busy
        if task_lock.is_user_busy(user_id):
            active_task = task_lock.get_active_task(user_id)
            await update.message.reply_text(
                f"⏳ Please wait! You have an active task: *{active_task}*",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        if not task_lock.lock_user(user_id, "Decryption"):
            return
        
        password, text = context.args[0], " ".join(context.args[1:])
        loading_msg = await update.message.reply_text("🔓 Decrypting...")
        
        fernet = Fernet(get_key_from_password(password))
        decrypted = fernet.decrypt(text.encode())
        await loading_msg.edit_text(
            f"✅ **Decrypted Text:**\n\n`{decrypted.decode()}`",
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception:
        await update.message.reply_text("❌ Decryption failed. Wrong password or invalid encrypted text.")
    finally:
        task_lock.unlock_user(user_id)

async def download_social_media(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str, platform: str):
    """Download videos/photos from Instagram, Facebook, etc."""
    loading_msg = await update.message.reply_text(f"⏳ Downloading from {platform}... Please wait...")
    filename = ""
    
    try:
        # Common options for yt-dlp
        ydl_opts = {
            'outtmpl': '%(id)s.%(ext)s',
            'noplaylist': True,
            'format': 'best',
        }
        
        # Add ffmpeg location if available
        if os.path.exists(FFMPEG_PATH):
            ydl_opts['ffmpeg_location'] = os.path.dirname(FFMPEG_PATH)
        
        await loading_msg.edit_text(f"📥 Downloading from {platform}...")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
        
        await loading_msg.edit_text("📤 Uploading file... This may take a while...")
        
        # Check file size - increased to 1GB (1024 MB)
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        if file_size_mb > 1024:
            await loading_msg.edit_text(
                f"❌ File too large ({file_size_mb:.1f}MB).\n\n"
                f"📦 Maximum allowed: 1GB (1024MB)"
            )
            if filename and os.path.exists(filename):
                os.remove(filename)
            return
        
        with open(filename, 'rb') as file:
            # Determine if it's a video or photo
            ext = filename.split('.')[-1].lower()
            if ext in ['jpg', 'jpeg', 'png', 'gif']:
                await context.bot.send_photo(
                    chat_id=update.message.chat_id,
                    photo=file,
                    caption=f"✅ Downloaded from {platform}!\n📦 Size: {file_size_mb:.1f}MB",
                    read_timeout=300,
                    write_timeout=300
                )
            else:
                await context.bot.send_video(
                    chat_id=update.message.chat_id,
                    video=file,
                    caption=f"✅ Downloaded from {platform}!\n📦 Size: {file_size_mb:.1f}MB",
                    read_timeout=300,
                    write_timeout=300,
                    supports_streaming=True
                )
        
        await loading_msg.edit_text(f"✅ Download complete from {platform}!")
    except asyncio.TimeoutError:
        logger.error(f"Timeout uploading file from {platform}")
        await loading_msg.edit_text("❌ Upload timed out. File may be too large.")
    except Exception as e:
        logger.error(f"Error downloading from {platform}: {e}")
        await loading_msg.edit_text(f"❌ Download failed. Please check the URL and try again.\n\n💡 Make sure the post is public!")
    finally:
        if filename and os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                pass

async def ytdl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    # Check if user is verified
    if not is_user_verified(user_id):
        await update.message.reply_text(
            "🔒 **Verification Required**\n\n"
            "Please use /start to verify your account first.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    url = " ".join(context.args)
    if not url:
        await update.message.reply_text(
            "❌ **Usage:** `/ytdl <URL>`\n\n"
            "💡 **Example:** `/ytdl https://youtu.be/xxxxx`\n\n"
            "🎥 **Supported:** YouTube, TikTok, Instagram, Facebook, Twitter, and 1000+ sites!",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Enhanced options with better quality control
    keyboard = [
        [
            InlineKeyboardButton("🎵 Audio (MP3)", callback_data="ytdl_audio"),
            InlineKeyboardButton("🎧 Audio (Best)", callback_data="ytdl_audio_hq")
        ],
        [
            InlineKeyboardButton("📱 Video 360p", callback_data="ytdl_360"),
            InlineKeyboardButton("📺 Video 480p", callback_data="ytdl_480")
        ],
        [
            InlineKeyboardButton("🎬 Video 720p", callback_data="ytdl_720"),
            InlineKeyboardButton("🌟 Best Quality", callback_data="ytdl_best")
        ],
        [InlineKeyboardButton("❌ Cancel", callback_data='main_menu')]
    ]
    await update.message.reply_text(
        '📺 **Advanced Video Downloader**\n\n'
        '🎯 **Choose your format:**\n\n'
        '🎵 **Audio MP3** - Fast, ~3-5MB\n'
        '🎧 **Audio Best** - High quality\n'
        '📱 **360p** - Small, ~20-40MB\n'
        '📺 **480p** - Good quality, ~40-80MB\n'
        '🎬 **720p** - HD quality, ~80-150MB\n'
        '🌟 **Best** - Highest quality available\n\n'
        '✨ **Powered by yt-dlp**\n'
        '🌐 Supports 1000+ websites!',
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN
    )

async def ytdl_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    
    # Check if user is already busy with another task
    if task_lock.is_user_busy(user_id):
        active_task = task_lock.get_active_task(user_id)
        await query.answer(
            f"⏳ Please wait! You have an active task: {active_task}",
            show_alert=True
        )
        return
    
    # Lock user for this download task
    if not task_lock.lock_user(user_id, "Video Download"):
        await query.answer("⏳ Please wait for your previous task to complete!", show_alert=True)
        return
    
    try:
        await query.answer()
        
        # Get download type from callback_data
        data = query.data.split('_', 2)
        download_type = data[1] if len(data) > 1 else 'auto'
        
        # Retrieve URL from storage instead of callback_data
        url = get_download_url(user_id)
        
        if not url:
            await query.edit_message_text(
                "❌ *Session Expired!*\n\n"
                "The download URL has expired or is not found.\n"
                "Please send the URL again and try once more.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("« Back", callback_data="menu_downloader")
                ]]),
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        # Enhanced quality labels
        quality_labels = {
            'audio': '🎵 Audio MP3',
            'audio_hq': '🎶 Audio HQ',
            '360': '📱 360p Video',
            '480': '📺 480p Video', 
            '720': '🎬 720p HD',
            'best': '🌟 Best Quality',
            'auto': '🎬 Auto Quality'
        }
        quality_label = quality_labels.get(download_type, '🎬 Video')
        
        # Animated loading messages
        loading_animations = [
            f"🔍 Analyzing video...",
            f"📊 Fetching metadata...",
            f"⚙️ Preparing download...",
            f"⏬ Downloading {quality_label}...",
            f"🎯 Processing {quality_label}...",
        ]
        
        loading_msg = await query.edit_message_text(text=loading_animations[0])
        filename = ""
        
        # Animation: Show analyzing
        await asyncio.sleep(0.5)
        await loading_msg.edit_text(loading_animations[1])
        
        # Create unique filename prefix for this user to prevent conflicts
        user_id = query.from_user.id
        timestamp = int(time.time())
        unique_prefix = f"{user_id}_{timestamp}"
        
        # Enhanced options for yt-dlp with anti-blocking measures and speed optimization
        common_opts = {
            'outtmpl': f'{unique_prefix}_%(title)s.%(ext)s',
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            # Anti-blocking measures
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'referer': 'https://www.google.com/',
            'headers': {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            },
            # Speed optimization - parallel downloads
            'concurrent_fragment_downloads': 8,  # Download 8 fragments simultaneously
            'http_chunk_size': 10485760,  # 10 MB chunks for faster downloads
            'buffersize': 16384,  # 16 KB buffer
            'throttledratelimit': None,  # No rate limiting
            # Retry and timeout settings
            'retries': 10,
            'fragment_retries': 10,
            'socket_timeout': 30,
            # Additional options
            'nocheckcertificate': True,
            'prefer_insecure': True,
            'geo_bypass': True,
            'geo_bypass_country': 'US',
            # Performance
            'noprogress': True,  # Disable progress bar for speed
            'no_color': True,
        }
        
        # Add ffmpeg location if available
        if os.path.exists(FFMPEG_PATH):
            common_opts['ffmpeg_location'] = os.path.dirname(FFMPEG_PATH)
        
        # Animation: Show preparing
        await asyncio.sleep(0.5)
        await loading_msg.edit_text(loading_animations[2])
        
        # Configure download based on type
        if download_type == 'audio' or download_type == 'audio_hq':
            # Audio download with animations
            await loading_msg.edit_text("🎵 Downloading audio track...")
            
            codec = 'mp3' if download_type == 'audio' else 'best'
            quality = '5' if download_type == 'audio_hq' else '7'  # VBR quality (0=best, 9=worst)
            
            ydl_opts = {
                **common_opts,
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': codec,
                    'preferredquality': quality,
                }],
            }
        else:
            # Video download with quality selection
            await loading_msg.edit_text(f"🎬 Downloading {quality_label}...")
            
            # Quality format strings optimized for file size
            format_strings = {
                '360': 'bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360]',
                '480': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]',
                '720': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]',
                'best': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'auto': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]'
            }
            
            format_string = format_strings.get(download_type, format_strings['auto'])
            
            ydl_opts = {
                **common_opts,
                'format': format_string,
                'merge_output_format': 'mp4',
            }
        
        # Animation: Show downloading
        await asyncio.sleep(0.5)
        await loading_msg.edit_text(loading_animations[3])
        
        # Progress tracking variables
        progress_data = {'status': '', 'percent': 0, 'speed': '', 'eta': ''}
        last_update_time = 0
        
        def progress_hook(d):
            """Real-time progress hook for yt-dlp"""
            nonlocal last_update_time, progress_data
            
            if d['status'] == 'downloading':
                # Extract progress info
                percent = d.get('_percent_str', '0%').strip()
                speed = d.get('_speed_str', 'N/A').strip()
                eta = d.get('_eta_str', 'N/A').strip()
                
                progress_data = {
                    'status': 'downloading',
                    'percent': percent,
                    'speed': speed,
                    'eta': eta
                }
            elif d['status'] == 'finished':
                progress_data['status'] = 'finished'
        
        # Add progress hook to options
        ydl_opts['progress_hooks'] = [progress_hook]
        
        # Download in thread pool (non-blocking for other users)
        loop = asyncio.get_event_loop()
        
        async def animate_download():
            """Show real-time progress bar while downloading"""
            nonlocal last_update_time
            
            while True:
                try:
                    current_time = time.time()
                    
                    # Update every 1 second for smoother progress
                    if current_time - last_update_time >= 1:
                        if progress_data['status'] == 'downloading':
                            # Create smooth progress bar (50 blocks for 2% increments)
                            try:
                                percent_val = float(progress_data['percent'].replace('%', ''))
                                filled = int(percent_val / 2)  # 50 blocks = 2% each
                                bar = '█' * filled + '░' * (50 - filled)
                                
                                msg = (
                                    f"📥 *Downloading {quality_label}*\n\n"
                                    f"Progress: `{bar}` {progress_data['percent']}\n"
                                    f"Speed: {progress_data['speed']}\n"
                                    f"ETA: {progress_data['eta']}"
                                )
                            except:
                                msg = f"📥 Downloading {quality_label}... {progress_data['percent']}"
                            
                            try:
                                await loading_msg.edit_text(msg, parse_mode=ParseMode.MARKDOWN)
                                last_update_time = current_time
                            except Exception as e:
                                # Ignore rate limit errors
                                if "message is not modified" not in str(e).lower():
                                    logger.debug(f"Progress update error: {e}")
                        
                        elif progress_data['status'] == 'finished':
                            await loading_msg.edit_text("✅ Download complete! Processing...")
                            break
                    
                    await asyncio.sleep(0.3)  # Check every 300ms for smoother updates
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Animation error: {e}")
                    await asyncio.sleep(1)
        
        async def animate_download_old():
            """Show animated progress while downloading (fallback)"""
            dots = ["⚫⚪⚪", "⚪⚫⚪", "⚪⚪⚫", "⚪⚫⚪"]
            progress_msgs = [
                "📥 Downloading... {dots}",
                "🔄 Processing video... {dots}",
                "⚙️ Encoding media... {dots}",
                "🎯 Finalizing... {dots}"
            ]
            idx = 0
            while True:
                try:
                    msg = progress_msgs[idx % len(progress_msgs)]
                    dot = dots[idx % len(dots)]
                    await loading_msg.edit_text(msg.format(dots=dot))
                    await asyncio.sleep(2)
                    idx += 1
                except:
                    break
        
        def download_with_ytdlp(url, opts, is_audio):
            """CPU-intensive YouTube download with yt-dlp - with 403 handling"""
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(info)
                    
                    # For audio, adjust extension
                    if is_audio:
                        base = filename.rsplit('.', 1)[0]
                        ext = 'mp3' if download_type == 'audio' else info.get('ext', 'mp3')
                        filename = f"{base}.{ext}"
                    
                    # Get video info
                    title = info.get('title', 'Video')
                    duration = info.get('duration', 0)
                    uploader = info.get('uploader', 'Unknown')
                    
                    return filename, {
                        'title': title,
                        'duration': duration,
                        'uploader': uploader
                    }
            except yt_dlp.utils.DownloadError as e:
                error_str = str(e)
                # If 403 error, try with additional options
                if '403' in error_str or 'Forbidden' in error_str:
                    logger.info("Got 403 error, retrying with additional options...")
                    
                    # Add more aggressive anti-blocking measures
                    opts['http_headers'] = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Sec-Fetch-User': '?1',
                        'Cache-Control': 'max-age=0',
                    }
                    opts['extractor_args'] = {'youtube': {'player_client': ['android', 'web']}}
                    
                    # Retry with new options
                    try:
                        with yt_dlp.YoutubeDL(opts) as ydl:
                            info = ydl.extract_info(url, download=True)
                            filename = ydl.prepare_filename(info)
                            
                            if is_audio:
                                base = filename.rsplit('.', 1)[0]
                                ext = 'mp3' if download_type == 'audio' else info.get('ext', 'mp3')
                                filename = f"{base}.{ext}"
                            
                            title = info.get('title', 'Video')
                            duration = info.get('duration', 0)
                            uploader = info.get('uploader', 'Unknown')
                            
                            return filename, {
                                'title': title,
                                'duration': duration,
                                'uploader': uploader
                            }
                    except Exception as retry_error:
                        logger.error(f"Retry with enhanced options also failed: {retry_error}")
                        raise e  # Raise original error
                else:
                    raise
            except Exception as e:
                logger.error(f"yt-dlp download error: {e}")
                raise
        
        # Start animation and download simultaneously
        animation_task = asyncio.create_task(animate_download())
        
        try:
            filename, video_info = await loop.run_in_executor(
                executor,
                download_with_ytdlp,
                url,
                ydl_opts,
                download_type in ['audio', 'audio_hq']
            )
        finally:
            animation_task.cancel()
            try:
                await animation_task
            except asyncio.CancelledError:
                pass
        
        # Animation: Show uploading
        await loading_msg.edit_text("📤 Preparing upload...")
        
        # Check file size
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        original_file_size_mb = file_size_mb  # Store original for percentage calculation
        
        # Warn user about very large files
        if file_size_mb > 100:
            await loading_msg.edit_text(
                f"⚠️ **Large File Detected!**\n\n"
                f"📊 Size: {file_size_mb:.1f} MB\n\n"
                f"This will take a while to upload.\n"
                f"Consider using a lower quality option for faster results.\n\n"
                f"📤 Preparing upload..."
            )
            await asyncio.sleep(3)
        
        # Handle large files with compression/splitting
        files_to_send = [filename]
        compressed = False
        split = False
        
        # Check Telegram limits
        if file_size_mb > 2000:  # Over 2GB (Telegram limit)
            await loading_msg.edit_text(
                f"❌ **File Too Large!**\n\n"
                f"📊 Size: {file_size_mb:.1f} MB\n"
                f"⚠️ Telegram Limit: 2000 MB\n\n"
                f"💡 **Try:**\n"
                f"• Lower quality (360p/480p)\n"
                f"• Audio only option\n"
                f"• Shorter video clip"
            )
            try:
                os.remove(filename)
            except:
                pass
            return
        
        # ⚠️ COMPRESSION DISABLED - Too slow!
        # Files >45MB will be split into chunks instead (instant!)
        
        # For videos: Try to upload up to 2GB directly (Telegram's real limit)
        # Only split if absolutely necessary (>2GB or upload fails)
        # Telegram has undocumented limits around 50MB for sendVideo
        # But we'll try anyway - worst case it fails and user can try lower quality
        if file_size_mb > 2000:  # Only split if over 2GB
            try:
                await loading_msg.edit_text(
                    f"✂️ **Splitting Large File...**\n\n"
                    f"📊 Size: {file_size_mb:.1f} MB\n"
                    f"📦 Creating 45MB chunks...\n\n"
                    f"⏳ Please wait..."
                )
                
                # Run split in thread pool
                loop = asyncio.get_event_loop()
                chunk_files = await loop.run_in_executor(None, split_file, filename, 45)
                
                if len(chunk_files) > 1:
                    files_to_send = chunk_files
                    split = True
                    
                    await loading_msg.edit_text(
                        f"✅ **File Split Complete!**\n\n"
                        f"📦 Created {len(chunk_files)} parts\n"
                        f"📊 ~{file_size_mb/len(chunk_files):.1f} MB each\n\n"
                        f"📤 Uploading parts..."
                    )
            except Exception as e:
                logger.error(f"File splitting failed: {e}")
                files_to_send = [filename]
        
        # Check Telegram limits one more time after processing
        for check_file in files_to_send:
            check_size = os.path.getsize(check_file) / (1024 * 1024)
            if check_size > 2000:
                await loading_msg.edit_text(
                    f"❌ **File Still Too Large!**\n\n"
                    f"📊 Size: {check_size:.1f} MB\n\n"
                    f"**Please use lower quality option.**"
                )
                for f in files_to_send:
                    try:
                        os.remove(f)
                    except:
                        pass
                return
        
        # Show upload progress with size info
        if file_size_mb > 50:
            upload_emoji = "🚀"
            warning = "\n⏳ Large file - Uploading may take time..."
            estimate_time = int(file_size_mb / 2)  # Rough estimate: 2MB/sec
            if estimate_time > 60:
                estimate_str = f"\n⏱️ Estimated: ~{estimate_time // 60} min"
            else:
                estimate_str = f"\n⏱️ Estimated: ~{estimate_time} sec"
        else:
            upload_emoji = "📤"
            warning = ""
            estimate_str = ""
        
        upload_start_time = time.time()
        await loading_msg.edit_text(
            f"{upload_emoji} **Uploading to Telegram...**\n\n"
            f"📊 Size: {file_size_mb:.1f} MB\n"
            f"🎯 Quality: {quality_label}{warning}{estimate_str}\n\n"
            f"💡 File will be auto-deleted after upload",
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Prepare caption with video info
        duration_str = ""
        if video_info.get('duration'):
            mins = int(video_info['duration'] // 60)
            secs = int(video_info['duration'] % 60)
            duration_str = f" • ⏱️ {mins}:{secs:02d}"
        
        # Use plain text to avoid Markdown parsing errors with special characters
        title = video_info.get('title', 'Video')[:50]
        uploader = video_info.get('uploader', 'Unknown')[:30]
        
        caption = (
            f"✅ Downloaded Successfully!\n\n"
            f"🎬 {title}\n"
            f"👤 {uploader}\n"
            f"📊 {file_size_mb:.1f} MB{duration_str}"
        )
        
        # Create upload progress animation
        async def animate_upload_progress():
            """Show animated upload progress"""
            progress_bars = [
                "▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱ 0%",
                "▰▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱ 5%",
                "▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱ 10%",
                "▰▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱ 15%",
                "▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱ 20%",
                "▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱ 25%",
                "▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱▱ 30%",
                "▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱ 35%",
                "▰▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱▱ 40%",
                "▰▰▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱ 45%",
                "▰▰▰▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱ 50%",
                "▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱ 55%",
                "▰▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱ 60%",
                "▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱▱▱▱ 65%",
                "▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱▱▱ 70%",
                "▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱▱ 75%",
                "▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱ 80%",
                "▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱▱ 85%",
                "▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱ 90%",
                "▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱ 95%",
                "▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰ 100%",
            ]
            idx = 0
            last_update = time.time()
            while True:
                try:
                    current_time = time.time()
                    elapsed = int(current_time - upload_start_time)
                    
                    # Update every 1.5 seconds for smoother progress
                    if current_time - last_update >= 1.5:
                        progress = progress_bars[min(idx, len(progress_bars) - 1)]
                        await loading_msg.edit_text(
                            f"📤 **Uploading...**\n\n"
                            f"{progress}\n\n"
                            f"📊 {file_size_mb:.1f} MB\n"
                            f"⏱️ Elapsed: {elapsed}s\n\n"
                            f"🗑️ Auto-cleanup enabled",
                            parse_mode=ParseMode.MARKDOWN
                        )
                        last_update = current_time
                        idx += 1
                    await asyncio.sleep(1)
                except:
                    break
        
        # Upload with animated progress
        upload_success = False
        # Increase retries for large files (they're more prone to network issues)
        max_retries = 3 if file_size_mb > 50 else 2
        retry_count = 0
        
        # Start upload animation
        animation_task = asyncio.create_task(animate_upload_progress())
        
        try:
            while retry_count < max_retries and not upload_success:
                try:
                    # Upload all files (single or split parts)
                    for idx, file_to_upload in enumerate(files_to_send):
                        part_caption = caption
                        
                        # Add part info if split
                        if split and len(files_to_send) > 1:
                            part_caption = f"📦 Part {idx + 1}/{len(files_to_send)}\n\n{caption}"
                        
                        # Send based on type
                        if download_type in ['audio', 'audio_hq']:
                            with open(file_to_upload, 'rb') as audio_file:
                                await context.bot.send_audio(
                                    chat_id=query.message.chat_id,
                                    audio=audio_file,
                                    caption=part_caption if idx == 0 else f"📦 Part {idx + 1}/{len(files_to_send)}",
                                    title=video_info.get('title', 'Audio'),
                                    performer=video_info.get('uploader', 'Unknown'),
                                    duration=video_info.get('duration', 0) if idx == 0 else None,
                                    write_timeout=3600,
                                    read_timeout=3600,
                                    connect_timeout=60,
                                    pool_timeout=60
                                )
                        elif split:
                            # Send split parts as documents
                            with open(file_to_upload, 'rb') as doc_file:
                                await context.bot.send_document(
                                    chat_id=query.message.chat_id,
                                    document=doc_file,
                                    caption=part_caption,
                                    write_timeout=3600,
                                    read_timeout=3600,
                                    connect_timeout=60,
                                    pool_timeout=60
                                )
                        else:
                            # For files >50MB, upload to Google Drive instead of Telegram
                            # Google Drive can handle files up to 15GB and provides shareable links
                            if file_size_mb > 50:
                                # Stop the upload animation
                                animation_task.cancel()
                                
                                # Progress tracking for Google Drive upload
                                progress_info = {'percent': 0, 'uploaded_mb': 0, 'total_mb': file_size_mb}
                                last_progress_update = 0
                                
                                async def update_drive_progress():
                                    """Update Google Drive upload progress"""
                                    nonlocal last_progress_update
                                    
                                    while True:
                                        try:
                                            current_time = time.time()
                                            
                                            # Update every 1 second for smoother progress
                                            if current_time - last_progress_update >= 1:
                                                percent = progress_info['percent']
                                                uploaded = progress_info['uploaded_mb']
                                                total = progress_info['total_mb']
                                                
                                                # Create smooth progress bar (50 blocks for 2% increments)
                                                filled = int(percent / 2)  # 50 blocks = 2% each
                                                bar = '█' * filled + '░' * (50 - filled)
                                                
                                                msg = (
                                                    f"☁️ **Uploading to Google Drive**\n\n"
                                                    f"Progress: `{bar}` {percent:.1f}%\n"
                                                    f"Uploaded: {uploaded:.1f} MB / {total:.1f} MB\n"
                                                    f"Quality: {quality_label}\n\n"
                                                    f"💡 Large files use Google Drive for reliability"
                                                )
                                                
                                                try:
                                                    await loading_msg.edit_text(msg, parse_mode=ParseMode.MARKDOWN)
                                                    last_progress_update = current_time
                                                except Exception as e:
                                                    if "message is not modified" not in str(e).lower():
                                                        logger.debug(f"Progress update error: {e}")
                                            
                                            await asyncio.sleep(0.3)  # Check every 300ms for smoother updates
                                        except asyncio.CancelledError:
                                            break
                                        except Exception as e:
                                            logger.error(f"Drive progress animation error: {e}")
                                            await asyncio.sleep(1)
                                
                                # Start progress animation
                                progress_task = asyncio.create_task(update_drive_progress())
                                
                                # Upload to Google Drive with progress callback
                                def drive_progress_callback(current, total):
                                    """Callback for Google Drive upload progress"""
                                    progress_info['percent'] = (current / total) * 100 if total > 0 else 0
                                    progress_info['uploaded_mb'] = current / (1024 * 1024)
                                    progress_info['total_mb'] = total / (1024 * 1024)
                                
                                # Upload in thread pool (blocking operation)
                                loop = asyncio.get_event_loop()
                                drive_link = await loop.run_in_executor(
                                    executor,
                                    upload_to_google_drive,
                                    file_to_upload,
                                    os.path.basename(file_to_upload),
                                    drive_progress_callback
                                )
                                
                                # Stop progress animation
                                progress_task.cancel()
                                try:
                                    await progress_task
                                except asyncio.CancelledError:
                                    pass
                                
                                if drive_link:
                                    # Send direct Google Drive link to all users
                                    final_link = drive_link
                                    link_type = "Direct Download Link"
                                    logger.info(f"Sending direct Google Drive link to user")
                                    
                                    # Extract file ID from the original link
                                    file_id = drive_link.split('id=')[1].split('&')[0] if 'id=' in drive_link else None
                                    
                                    # Send success message with download link
                                    message_text = (
                                        f"✅ **Video Ready!**\n\n"
                                        f"🎬 {title}\n"
                                        f"👤 {uploader}\n"
                                        f"📊 {file_size_mb:.1f} MB{duration_str}\n\n"
                                        f"📥 **{link_type}:**\n"
                                        f"{final_link}\n\n"
                                        f"💡 Click link to download\n"
                                        f"🔒 Link never expires"
                                    )
                                    
                                    await context.bot.send_message(
                                        chat_id=query.message.chat_id,
                                        text=message_text,
                                        parse_mode=ParseMode.MARKDOWN
                                    )
                                    upload_success = True
                                else:
                                    # Google Drive upload failed - notify user
                                    await loading_msg.edit_text(
                                        f"❌ **Upload Failed**\n\n"
                                        f"Google Drive upload encountered an error.\n\n"
                                        f"💡 **Possible reasons:**\n"
                                        f"• Google Drive not configured\n"
                                        f"• Authentication expired\n"
                                        f"• Storage quota exceeded\n\n"
                                        f"Please contact admin or try a lower quality.",
                                        parse_mode=ParseMode.MARKDOWN
                                    )
                            else:
                                # Regular video upload for files ≤50MB
                                with open(file_to_upload, 'rb') as video_file:
                                    await context.bot.send_video(
                                        chat_id=query.message.chat_id,
                                        video=video_file,
                                        caption=part_caption,
                                        duration=video_info.get('duration', 0),
                                        supports_streaming=True,
                                        write_timeout=7200,  # Increased timeout
                                        read_timeout=7200,
                                        connect_timeout=120,
                                        pool_timeout=120
                                    )
                        
                        # Small delay between parts
                        if len(files_to_send) > 1 and idx < len(files_to_send) - 1:
                            await asyncio.sleep(1)

                    
                    upload_success = True
                    upload_time = int(time.time() - upload_start_time)
                    
                    # Stop animation first to avoid race condition
                    animation_task.cancel()
                    try:
                        await animation_task
                    except asyncio.CancelledError:
                        pass
                    
                    # Show success message with main menu
                    await loading_msg.edit_text(
                        f"✅ **Download Complete!** 🎉\n\n"
                        f"⏱️ Upload time: {upload_time}s\n\n"
                        f"🎯 Returning to main menu...",
                        reply_markup=get_main_menu_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    
                    # Small delay then automatically show main menu
                    await asyncio.sleep(1.5)
                    await loading_msg.edit_text(
                        "🤖 *Welcome to Super Bot!*\n\n"
                        "Choose a feature from the menu below:",
                        reply_markup=get_main_menu_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    
                except Exception as upload_error:
                    retry_count += 1
                    error_name = type(upload_error).__name__
                    logger.error(f"Upload attempt {retry_count} failed: {error_name}")
                    
                    if retry_count < max_retries:
                        await loading_msg.edit_text(
                            f"⚠️ Retry {retry_count}/{max_retries}...\n"
                            f"📤 Uploading again..."
                        )
                        await asyncio.sleep(2)
                    else:
                        # Stop animation before showing error
                        animation_task.cancel()
                        try:
                            await animation_task
                        except asyncio.CancelledError:
                            pass
                        raise upload_error
        finally:
            # Ensure animation is stopped if still running
            if not animation_task.cancelled() and not animation_task.done():
                animation_task.cancel()
                try:
                    await animation_task
                except asyncio.CancelledError:
                    pass
                    
    except asyncio.TimeoutError:
        logger.error(f"Timeout uploading file for user {query.from_user.id}")
        await loading_msg.edit_text(
            "❌ **Upload Timeout**\n\n"
            "⏱️ Upload took too long\n\n"
            "**Try:**\n"
            "• Lower quality option\n"
            "• Audio only format\n"
            "• Shorter video duration\n"
            "• Try again with better connection",
            reply_markup=get_main_menu_keyboard()
        )
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp download error for user {query.from_user.id}: {e}")
        error_msg = str(e).lower()
        
        # Provide specific error messages based on error type
        if '403' in error_msg or 'forbidden' in error_msg:
            message = (
                "❌ **Access Blocked (403)**\n\n"
                "The website blocked the download request.\n\n"
                "**This usually happens when:**\n"
                "• Video has DRM protection\n"
                "• Site requires login/subscription\n"
                "• Geographic restrictions\n"
                "• Anti-bot measures active\n\n"
                "**Try:**\n"
                "✅ Different video from same site\n"
                "✅ Lower quality option\n"
                "✅ Audio only format\n"
                "✅ Wait a few minutes and retry\n\n"
                "💡 Some premium/protected content cannot be downloaded."
            )
        elif 'private' in error_msg or 'unavailable' in error_msg:
            message = (
                "❌ **Video Unavailable**\n\n"
                "This video cannot be accessed.\n\n"
                "**Possible reasons:**\n"
                "• Video is private or removed\n"
                "• Account required\n"
                "• Geographic restrictions"
            )
        elif 'age' in error_msg or 'sign in' in error_msg:
            message = (
                "❌ **Age-Restricted Content**\n\n"
                "This video requires sign-in.\n\n"
                "Bot cannot download age-restricted content."
            )
        elif 'format' in error_msg:
            message = (
                "❌ **Format Not Available**\n\n"
                "Requested quality not available.\n\n"
                "**Try:**\n"
                "• Different quality option\n"
                "• Audio only format"
            )
        elif '429' in error_msg or 'too many' in error_msg:
            message = (
                "❌ **Rate Limited**\n\n"
                "Too many requests detected.\n\n"
                "**Please:**\n"
                "⏰ Wait 5-10 minutes\n"
                "🔄 Then try again"
            )
        else:
            message = (
                "❌ **Download Failed**\n\n"
                "Could not download this video.\n\n"
                "**Try:**\n"
                "• Different video\n"
                "• Different quality\n"
                "• Check if URL is valid\n"
                "• Try audio format instead"
            )
        
        await loading_msg.edit_text(message, reply_markup=get_main_menu_keyboard())
    except Exception as e:
        error_name = type(e).__name__
        error_msg = str(e)
        logger.error(f"Error in ytdl_callback for user {query.from_user.id}: {error_name}: {error_msg}")
        
        # Smart error messages
        if 'storagequotaexceeded' in error_msg.lower() or 'service accounts do not have storage' in error_msg.lower():
            # Service account storage issue
            message = (
                "❌ **Google Drive Setup Error**\n\n"
                f"📊 File size: {file_size_mb:.1f} MB\n\n"
                "⚠️ **Service accounts don't have storage quota!**\n\n"
                "**Solution:**\n"
                "📋 Admin needs to create OAuth2 credentials\n"
                "📖 See: OAUTH2_QUICK_FIX.md\n\n"
                "**Workaround (for now):**\n"
                "✅ Use 480p quality (~30-40MB)\n"
                "✅ Use 360p quality (~20-25MB)\n"
                "✅ Use Audio Only format\n\n"
                "💡 Files under 50MB upload directly to Telegram!"
            )
        elif 'network' in error_msg.lower() or 'connection' in error_msg.lower() or 'readerror' in error_msg.lower():
            # For large files, this might be a Google Drive issue
            if file_size_mb > 50:
                message = (
                    "❌ **Upload Failed**\n\n"
                    f"📊 File size: {file_size_mb:.1f} MB\n"
                    "⚠️ Google Drive upload failed\n\n"
                    "**Possible reasons:**\n"
                    "• Google Drive not configured correctly\n"
                    "• Network connection interrupted\n"
                    "• Storage quota exceeded\n\n"
                    "**Recommendations:**\n"
                    "✅ Contact admin to check Google Drive\n"
                    "✅ Try lower quality (480p/360p)\n"
                    "✅ Use **Audio Only** format\n\n"
                    "💡 Files under 50MB upload directly to Telegram!"
                )
            else:
                message = (
                    "❌ **Network Error**\n\n"
                    "Connection issue occurred.\n\n"
                    "**Please:**\n"
                    "• Check your internet\n"
                    "• Try again in a moment"
                )
        elif file_size_mb > 1000:
            message = (
                "❌ **File Too Large**\n\n"
                f"📊 Size: {file_size_mb:.1f} MB\n"
                f"⚠️ Recommended: < 1000 MB\n\n"
                "**Solutions:**\n"
                "✅ Choose 360p or 480p\n"
                "✅ Use Audio Only\n"
                "✅ Download shorter videos"
            )
        else:
            message = (
                "❌ **Upload Failed**\n\n"
                f"📊 File size: {file_size_mb:.1f} MB\n"
                f"⚠️ {error_name}\n\n"
                "**Recommendations:**\n"
                "✅ Try lower quality (360p/480p)\n"
                "✅ Use Audio Only option\n"
                "✅ Videos < 100MB work best"
            )
        
        await loading_msg.edit_text(message, reply_markup=get_main_menu_keyboard())
    finally:
        # Unlock user after task completion
        task_lock.unlock_user(user_id)
        
        # IMMEDIATE file cleanup - handle all files (original + split parts)
        all_files_to_clean = []
        
        # Add main file
        if filename and os.path.exists(filename):
            all_files_to_clean.append(filename)
        
        # Add split parts if any
        if files_to_send and len(files_to_send) > 1:
            for part_file in files_to_send:
                if part_file != filename and os.path.exists(part_file):
                    all_files_to_clean.append(part_file)
        
        # Clean up all files
        for file_to_clean in all_files_to_clean:
            cleanup_attempts = 0
            max_cleanup_attempts = 3
            
            while cleanup_attempts < max_cleanup_attempts:
                try:
                    # Close any file handles
                    import gc
                    gc.collect()
                    
                    # Try to delete immediately
                    os.remove(file_to_clean)
                    logger.info(f"✅ Cleaned up temporary file: {file_to_clean}")
                    break
                    
                except PermissionError as e:
                    cleanup_attempts += 1
                    logger.warning(f"File locked, retry {cleanup_attempts}/{max_cleanup_attempts}: {file_to_clean}")
                    if cleanup_attempts < max_cleanup_attempts:
                        await asyncio.sleep(1)
                    else:
                        # Last resort: schedule for deletion
                        logger.error(f"❌ Could not delete {file_to_clean} after {max_cleanup_attempts} attempts")
                        try:
                            # Try to delete in background
                            def delayed_cleanup(file_path):
                                for i in range(5):
                                    try:
                                        time.sleep(2)
                                        if os.path.exists(file_path):
                                            os.remove(file_path)
                                            logger.info(f"✅ Delayed cleanup successful: {file_path}")
                                        break
                                    except:
                                        continue
                            
                            import threading
                            cleanup_thread = threading.Thread(target=delayed_cleanup, args=(file_to_clean,))
                            cleanup_thread.daemon = True
                            cleanup_thread.start()
                        except Exception as bg_error:
                            logger.error(f"Background cleanup failed: {bg_error}")
                            
                except Exception as e:
                    logger.error(f"Unexpected error during cleanup of {file_to_clean}: {e}")
                    break

async def shorten_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    url = " ".join(context.args)
    if not url:
        await update.message.reply_text(
            "❌ **Usage:** `/shorten <URL>`\n\n"
            "💡 **Example:** `/shorten https://example.com/very/long/url`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Check if user is busy
    if task_lock.is_user_busy(user_id):
        active_task = task_lock.get_active_task(user_id)
        await update.message.reply_text(
            f"⏳ Please wait! You have an active task: *{active_task}*",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    if not task_lock.lock_user(user_id, "URL Shortening"):
        return
    
    loading_msg = await update.message.reply_text("🔗 Shortening URL...")
    try:
        # Use TinyURL shortener with updated API
        shortener = pyshorteners.Shortener()
        short_url = shortener.tinyurl.short(url)
        await loading_msg.edit_text(
            f"✅ **URL Shortened!**\n\n"
            f"🔗 **Original:** `{url}`\n"
            f"✨ **Shortened:** {short_url}",
            parse_mode=ParseMode.MARKDOWN
        )
    except AttributeError:
        # Fallback: If tinyurl doesn't work, try a simple approach
        try:
            import urllib.parse
            import urllib.request
            encoded_url = urllib.parse.quote(url, safe='')
            api_url = f"https://tinyurl.com/api-create.php?url={encoded_url}"
            with urllib.request.urlopen(api_url, timeout=10) as response:
                short_url = response.read().decode('utf-8')
            await loading_msg.edit_text(
                f"✅ **URL Shortened!**\n\n"
                f"🔗 **Original:** `{url}`\n"
                f"✨ **Shortened:** {short_url}",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            logger.error(f"URL shortening error: {e}")
            await loading_msg.edit_text("❌ Could not shorten URL. Please check the URL and try again.")
    except Exception as e:
        logger.error(f"URL shortening error: {e}")
        await loading_msg.edit_text("❌ Could not shorten URL. Please check the URL and try again.")
    finally:
        task_lock.unlock_user(user_id)

async def qr_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = " ".join(context.args)
    if not text:
        await update.message.reply_text(
            "❌ **Usage:** `/qr <text or URL>`\n\n"
            "💡 **Examples:**\n"
            "`/qr https://example.com`\n"
            "`/qr My contact info`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Check if user is busy
    if task_lock.is_user_busy(user_id):
        active_task = task_lock.get_active_task(user_id)
        await update.message.reply_text(
            f"⏳ Please wait! You have an active task: *{active_task}*",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    if not task_lock.lock_user(user_id, "QR Generation"):
        return
    
    loading_msg = await update.message.reply_text("📱 Generating QR code...")
    try:
        # Generate QR code
        buf = io.BytesIO()
        qr_img = qrcode.make(text)
        qr_img.save(buf, 'PNG')
        buf.seek(0)
        buf.name = 'qrcode.png'
        
        # Send QR code
        await update.message.reply_photo(
            photo=buf, 
            caption=f"✅ QR Code generated!\n\n📝 Content: {text[:50]}{'...' if len(text) > 50 else ''}"
        )
        await loading_msg.delete()
    except Exception as e:
        logger.error(f"QR code generation error: {e}")
        await loading_msg.edit_text("❌ Failed to generate QR code. Please try again.")
    finally:
        task_lock.unlock_user(user_id)
async def addnote_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    note_text = " ".join(context.args)
    if not note_text:
        await update.message.reply_text(
            "❌ **Usage:** `/addnote [note]`\n\n"
            "💡 **Example:** `/addnote Buy groceries tomorrow`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    loading_msg = await update.message.reply_text("💾 Saving note...")
    with sqlite3.connect('bot_database.db') as conn:
        conn.cursor().execute("INSERT INTO notes (user_id, note) VALUES (?, ?)", (user_id, note_text))
        conn.commit()
    await loading_msg.edit_text("✅ Note saved successfully!")

async def notes_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    loading_msg = await update.message.reply_text("📋 Loading your notes...")
    
    with sqlite3.connect('bot_database.db') as conn:
        notes = conn.cursor().execute(
            "SELECT id, note, created_at FROM notes WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
    
    if not notes:
        await loading_msg.edit_text("📝 You have no saved notes.\n\n💡 Use `/addnote [text]` to create one!", parse_mode=ParseMode.MARKDOWN)
        return
    
    response = "📋 **Your Notes:**\n\n"
    for note in notes:
        note_id, note_text, created_at = note
        date = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S').strftime('%d %b %Y')
        response += f"🔹 **ID:** `{note_id}` | 📅 {date}\n📝 {note_text}\n\n"
    
    response += "💡 Use `/delnote [id]` to delete a note"
    await loading_msg.edit_text(response, parse_mode=ParseMode.MARKDOWN)

async def delnote_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        note_id = int(context.args[0])
    except (IndexError, ValueError):
        await update.message.reply_text(
            "❌ **Usage:** `/delnote <id>`\n\n"
            "💡 Use `/notes` to see note IDs",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    loading_msg = await update.message.reply_text("🗑️ Deleting note...")
    with sqlite3.connect('bot_database.db') as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM notes WHERE id = ? AND user_id = ?", (note_id, user_id))
        if cursor.rowcount == 0:
            await loading_msg.edit_text(f"❌ Note ID `{note_id}` not found.", parse_mode=ParseMode.MARKDOWN)
        else:
            await loading_msg.edit_text(f"✅ Note ID `{note_id}` deleted successfully!", parse_mode=ParseMode.MARKDOWN)
        conn.commit()

async def password_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        length = int(context.args[0]) if context.args else 16
        if not 8 <= length <= 128:
            await update.message.reply_text(
                "❌ Password length must be between 8 and 128 characters.\n\n"
                "💡 **Example:** `/password 20`",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        # Check if user is busy
        if task_lock.is_user_busy(user_id):
            active_task = task_lock.get_active_task(user_id)
            await update.message.reply_text(
                f"⏳ Please wait! You have an active task: *{active_task}*",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        if not task_lock.lock_user(user_id, "Password Generation"):
            return
        
        loading_msg = await update.message.reply_text("🔑 Generating secure password...")
        alphabet = string.ascii_letters + string.digits + string.punctuation
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        
        await loading_msg.edit_text(
            f"✅ **Secure Password Generated!**\n\n"
            f"🔐 `{password}`\n\n"
            f"📏 Length: {length} characters\n"
            f"💡 Tap to copy!",
            parse_mode=ParseMode.MARKDOWN
        )
    except (ValueError, IndexError):
        await update.message.reply_text(
            "❌ Invalid length.\n\n"
            "💡 **Usage:** `/password [length]`\n"
            "**Example:** `/password 20`",
            parse_mode=ParseMode.MARKDOWN
        )
    finally:
        task_lock.unlock_user(user_id)

async def deletefile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Delete a stored file from Google Drive and database"""
    user_id = update.effective_user.id
    
    # Check if user is verified
    if not is_user_verified(user_id):
        await update.message.reply_text(
            "🔒 **Verification Required**\n\n"
            "Please use /start to verify your account first.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    try:
        file_id = int(context.args[0])
    except (IndexError, ValueError):
        await update.message.reply_text(
            "❌ **Usage:** `/deletefile <id>`\n\n"
            "💡 Use 'My Files' from File Storage menu to see file IDs",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    loading_msg = await update.message.reply_text("🗑️ Deleting file...")
    
    try:
        # Initialize Drive service
        drive_service = get_drive_service()
        # Get file info from database
        result = db_pool.execute(
            "SELECT filename, drive_file_id FROM stored_files WHERE id = ? AND user_id = ?",
            (file_id, user_id)
        )
        
        if not result:
            await loading_msg.edit_text(
                f"❌ File ID `{file_id}` not found.\n\n"
                "💡 Check your file IDs using 'My Files'",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        filename, drive_file_id = result[0]
        
        # Delete from Google Drive
        try:
            if drive_service and drive_file_id:
                drive_service.files().delete(fileId=drive_file_id).execute()
        except Exception as drive_error:
            logger.warning(f"Failed to delete from Drive (file may already be deleted): {drive_error}")
        
        # Delete from database
        db_pool.execute(
            "DELETE FROM stored_files WHERE id = ? AND user_id = ?",
            (file_id, user_id),
            fetch='none'
        )
        
        await loading_msg.edit_text(
            f"✅ **File Deleted Successfully!**\n\n"
            f"📄 **Filename:** `{filename}`\n"
            f"🗑️ Removed from storage\n\n"
            f"💡 Use 'My Files' to view remaining files",
            parse_mode=ParseMode.MARKDOWN
        )
        
    except Exception as e:
        logger.error(f"Error in deletefile_command: {e}")
        await loading_msg.edit_text(
            f"❌ **Failed to delete file**\n\n"
            f"Error: {str(e)}",
            parse_mode=ParseMode.MARKDOWN
        )


async def compress_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start image compression mode"""
    user_id = update.effective_user.id
    
    # Check if user is verified
    if not is_user_verified(user_id):
        await update.message.reply_text(
            "🔒 **Verification Required**\n\n"
            "Please use /start to verify your account first.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Get quality level
    quality_map = {
        'high': 85,
        'medium': 70,
        'low': 50
    }
    
    quality_name = context.args[0].lower() if context.args else 'high'
    quality = quality_map.get(quality_name, 85)
    
    # Set user state
    session_manager.set_state(user_id, 'awaiting_compress_image')
    context.user_data['compress_quality'] = quality
    context.user_data['compress_quality_name'] = quality_name
    
    await update.message.reply_text(
        f"🗜️ **Image Compressor**\n\n"
        f"📊 **Quality:** {quality_name.capitalize()} ({quality}%)\n\n"
        f"📸 **Now send me a photo to compress!**\n\n"
        f"💡 The file size will be reduced while maintaining quality.",
        parse_mode=ParseMode.MARKDOWN
    )


async def photopdf_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start photo to PDF collection mode"""
    user_id = update.effective_user.id
    
    # Check if user is verified
    if not is_user_verified(user_id):
        await update.message.reply_text(
            "🔒 **Verification Required**\n\n"
            "Please use /start to verify your account first.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Initialize photo collection
    session_manager.set_state(user_id, 'collecting_photos_for_pdf')
    context.user_data['pdf_photos'] = []
    
    await update.message.reply_text(
        "📸 **Photos to PDF**\n\n"
        "📚 **Started collecting photos!**\n\n"
        "📤 Send me photos one by one\n"
        "✅ Type `/done` when finished\n\n"
        "💡 All photos will be combined into a single PDF in the order you send them.",
        parse_mode=ParseMode.MARKDOWN
    )


async def done_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Finish collecting photos and create PDF"""
    user_id = update.effective_user.id
    
    # Check if user is in photo collection mode
    if session_manager.get_state(user_id) != 'collecting_photos_for_pdf':
        await update.message.reply_text(
            "❌ You're not collecting photos.\n\n"
            "💡 Use `/photopdf` first to start collecting photos.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    photos = context.user_data.get('pdf_photos', [])
    
    if not photos:
        await update.message.reply_text(
            "❌ No photos collected yet!\n\n"
            "📤 Send at least one photo before using `/done`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Clear state
    session_manager.clear_state(user_id)
    context.user_data['pdf_photos'] = []
    
    # Check if user is busy
    if task_lock.is_user_busy(user_id):
        active_task = task_lock.get_active_task(user_id)
        await update.message.reply_text(
            f"⏳ Please wait! You have an active task: *{active_task}*",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    if not task_lock.lock_user(user_id, "PDF Creation"):
        await update.message.reply_text("⚠️ Unable to start task. Please try again.")
        return
    
    loading_msg = await update.message.reply_text(
        f"📄 Creating PDF with {len(photos)} photo(s)..."
    )
    
    try:
        timestamp = int(time.time())
        pdf_path = f"{user_id}_{timestamp}_photos.pdf"
        
        await loading_msg.edit_text("📥 Downloading photos...")
        
        # Download all photos
        downloaded_paths = []
        for idx, photo_file_id in enumerate(photos):
            photo_path = f"{user_id}_{timestamp}_photo_{idx}.jpg"
            file = await context.bot.get_file(photo_file_id)
            await file.download_to_drive(photo_path)
            downloaded_paths.append(photo_path)
        
        await loading_msg.edit_text("📝 Creating PDF document...")
        
        # Create PDF in thread pool
        loop = asyncio.get_event_loop()
        
        def create_pdf_from_photos(photo_paths, output_path):
            """Create PDF from multiple photos"""
            c = canvas.Canvas(output_path, pagesize=A4)
            page_width, page_height = A4
            
            for photo_path in photo_paths:
                # Open image to get dimensions
                img = Image.open(photo_path)
                img_width, img_height = img.size
                
                # Calculate scaling to fit page with margin
                margin = 50
                available_width = page_width - (2 * margin)
                available_height = page_height - (2 * margin)
                
                # Scale to fit
                scale = min(available_width / img_width, available_height / img_height)
                scaled_width = img_width * scale
                scaled_height = img_height * scale
                
                # Center image on page
                x = (page_width - scaled_width) / 2
                y = (page_height - scaled_height) / 2
                
                # Draw image
                c.drawImage(photo_path, x, y, width=scaled_width, height=scaled_height)
                c.showPage()
            
            c.save()
        
        await loop.run_in_executor(executor, create_pdf_from_photos, downloaded_paths, pdf_path)
        
        await loading_msg.edit_text("📤 Uploading PDF...")
        
        # Send PDF
        with open(pdf_path, 'rb') as pdf_file:
            await update.message.reply_document(
                document=pdf_file,
                caption=f"✅ PDF created with {len(photos)} photo(s)!",
                filename=f"photos_{timestamp}.pdf"
            )
        
        # Show success with main menu
        await loading_msg.edit_text(
            f"✅ **PDF Created Successfully!**\n\n"
            f"📸 Photos: {len(photos)}\n"
            f"📄 Format: PDF\n\n"
            f"🎯 Returning to main menu...",
            reply_markup=get_main_menu_keyboard(),
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Auto-return to main menu
        await asyncio.sleep(1.5)
        await loading_msg.edit_text(
            "🤖 *Welcome to Super Bot!*\n\n"
            "Choose a feature from the menu below:",
            reply_markup=get_main_menu_keyboard(),
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Clean up
        for photo_path in downloaded_paths:
            if os.path.exists(photo_path):
                os.remove(photo_path)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
    except Exception as e:
        logger.error(f"Error in done_command: {e}")
        await loading_msg.edit_text(
            f"❌ Failed to create PDF: {str(e)}",
            reply_markup=get_main_menu_keyboard()
        )
    finally:
        task_lock.unlock_user(user_id)


# --- 7. Main Bot Function ---
def main():
    """Starts the bot and sets up all handlers."""
    # Configuration is already validated at startup
    init_db()
    
    # OPTIMIZED Application - Simple and reliable (without job queue to avoid timezone issues)
    from telegram.ext._applicationbuilder import ApplicationBuilder
    builder = ApplicationBuilder()
    builder.token(BOT_TOKEN)
    builder.concurrent_updates(True)
    builder.read_timeout(None)  # Unlimited for uploads
    builder.write_timeout(None)  # Unlimited for uploads
    builder.connect_timeout(60)
    builder.pool_timeout(60)
    builder.get_updates_read_timeout(30)
    builder.get_updates_write_timeout(30)
    builder.get_updates_connect_timeout(30)
    builder.get_updates_pool_timeout(30)
    
    # Explicitly disable job queue to avoid timezone issues
    try:
        from telegram.ext._utils.types import ODVInput
        from telegram.ext._utils.defaultvalue import DefaultValue, DEFAULT_NONE
        builder._job_queue = DEFAULT_NONE
    except:
        pass
    
    application = builder.build()

    # --- Add ALL handlers ---
    # Admin handlers
    application.add_handler(CommandHandler("admin", admin_panel))
    application.add_handler(CommandHandler("setupi", set_upi_command))
    application.add_handler(CommandHandler("setprice", set_price_command))
    application.add_handler(CommandHandler("setqr", set_qr_command))
    application.add_handler(CallbackQueryHandler(admin_button_callback, pattern='^admin_'))
    application.add_handler(CallbackQueryHandler(admin_button_callback, pattern='^payment_'))
    
    # User commands - Essential + New Features
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("account", account_command))
    application.add_handler(CommandHandler("premium", premium_command))
    
    # Tool commands
    application.add_handler(CommandHandler("ytdl", ytdl_command))
    application.add_handler(CommandHandler("shorten", shorten_command))
    application.add_handler(CommandHandler("qr", qr_command))
    application.add_handler(CommandHandler("tts", tts_command))
    application.add_handler(CommandHandler("translate", translate_command))
    application.add_handler(CommandHandler("encrypt", encrypt_command))
    application.add_handler(CommandHandler("decrypt", decrypt_command))
    application.add_handler(CommandHandler("addnote", addnote_command))
    application.add_handler(CommandHandler("notes", notes_command))
    application.add_handler(CommandHandler("password", password_command))
    application.add_handler(CommandHandler("delnote", delnote_command))
    application.add_handler(CommandHandler("deletefile", deletefile_command))
    application.add_handler(CommandHandler("compress", compress_command))
    application.add_handler(CommandHandler("photopdf", photopdf_command))
    application.add_handler(CommandHandler("done", done_command))
    application.add_handler(CommandHandler("drivetest", drivetest_command))

    # User button handlers
    application.add_handler(CallbackQueryHandler(button_callback_handler, pattern='^verify_user$'))
    application.add_handler(CallbackQueryHandler(button_callback_handler, pattern='^menu_'))
    application.add_handler(CallbackQueryHandler(button_callback_handler, pattern='^help_'))
    application.add_handler(CallbackQueryHandler(button_callback_handler, pattern='^action_'))
    application.add_handler(CallbackQueryHandler(button_callback_handler, pattern='^get_premium$'))
    application.add_handler(CallbackQueryHandler(button_callback_handler, pattern='^donate$'))
    application.add_handler(CallbackQueryHandler(button_callback_handler, pattern='^submit_payment$'))
    application.add_handler(CallbackQueryHandler(button_callback_handler, pattern='^my_account$'))
    # Missing handler for feature request button (callback_data='request_feature')
    application.add_handler(CallbackQueryHandler(button_callback_handler, pattern='^request_feature$'))
    # Handlers for new features
    application.add_handler(CallbackQueryHandler(button_callback_handler, pattern='^compress_'))
    application.add_handler(CallbackQueryHandler(button_callback_handler, pattern='^done_sending_photos$'))
    application.add_handler(CallbackQueryHandler(button_callback_handler, pattern='^create_pdf$'))
    application.add_handler(CallbackQueryHandler(button_callback_handler, pattern='^delete_stored_'))
    application.add_handler(CallbackQueryHandler(ytdl_callback, pattern='^ytdl_'))
    application.add_handler(CallbackQueryHandler(button_callback_handler, pattern='^main_menu$'))

    # Message and File handlers
    # IMPORTANT: Specific handlers MUST come BEFORE the general broadcast handler
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo))
    application.add_handler(MessageHandler(filters.VIDEO & ~filters.COMMAND, handle_video))
    application.add_handler(MessageHandler(filters.Document.PDF & ~filters.COMMAND, handle_pdf))
    application.add_handler(MessageHandler((filters.Document.ALL | filters.PHOTO | filters.VIDEO | filters.AUDIO) & ~filters.COMMAND, handle_file_upload))
    
    # Broadcast handler comes LAST to catch any remaining non-command messages for admin broadcast
    application.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_broadcast_message))

    # Post-init: Start background cleanup task
    async def post_init(app: Application):
        """Start background tasks after bot initialization"""
        asyncio.create_task(cleanup_old_downloads())
        logger.info("🗑️ Auto-cleanup task started")
    
    application.post_init = post_init
    
    print("Bot with Concurrent Processing is running...")
    print("Multiple users can now use the bot simultaneously!")
    print("All requests are processed in parallel for maximum speed.")
    print("🗑️ Auto-cleanup enabled: Old downloads removed every 5 minutes")
    application.run_polling()

if __name__ == '__main__':
    main()