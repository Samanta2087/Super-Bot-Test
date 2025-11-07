#!/bin/bash
# Setup script for Super Bot on Linux

echo "🤖 Super Bot Setup Script"
echo "========================="
echo ""

# Check if .env exists
if [ -f ".env" ]; then
    echo "✓ .env file already exists"
else
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file with your credentials:"
    echo "   nano .env"
    echo ""
    echo "   You need to set:"
    echo "   - BOT_TOKEN (from @BotFather)"
    echo "   - ADMIN_USER_ID (your Telegram user ID)"
    echo "   - GOOGLE credentials (if using Google Drive)"
    echo ""
fi

# Check FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo "✓ FFmpeg is installed"
else
    echo "⚠️  FFmpeg not found"
    echo "   Install with: sudo apt-get install ffmpeg -y"
fi

# Check Python version
echo ""
echo "Python version: $(python3 --version)"

# Install dependencies
echo ""
read -p "Install Python dependencies? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
fi

echo ""
echo "========================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file: nano .env"
echo "2. Add your BOT_TOKEN and ADMIN_USER_ID"
echo "3. Run bot: python3 super.py"
echo ""
