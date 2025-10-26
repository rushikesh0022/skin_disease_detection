#!/bin/bash

# Start Face Concern Detector Web Application
echo "🚀 Starting Face Concern Detector Web Application..."
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if Flask is installed
if ! python3 -c "import flask" &> /dev/null; then
    echo "⚠️  Flask not found. Installing..."
    pip3 install flask flask-cors
fi

echo "✅ Starting Flask API server..."
echo "📡 API will be available at: http://localhost:5000"
echo "🌐 Frontend will be available at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the servers"
echo "=================================================="

# Start Flask API in background
cd "$(dirname "$0")"
python3 app/flask_api.py &
API_PID=$!

# Wait a moment for API to start
sleep 3

# Start simple HTTP server for frontend
python3 -m http.server 8000 &
WEB_PID=$!

# Open browser (macOS)
if command -v open &> /dev/null; then
    sleep 2
    open http://localhost:8000/frontend.html
fi

# Wait for user to stop
echo ""
echo "🎉 Application is running!"
echo "📱 Open http://localhost:8000/frontend.html in your browser"
echo ""
echo "Press Enter to stop all servers..."
read

# Kill background processes
kill $API_PID 2>/dev/null
kill $WEB_PID 2>/dev/null

echo "👋 Stopped all servers. Goodbye!"
