#!/bin/bash

# Face Concern Detector - Quick Launcher
echo "🚀 Face Concern Detector - Quick Access"
echo "======================================="
echo ""
echo "✅ Flask API: http://localhost:5001 (Running)"
echo "✅ Frontend:  http://localhost:8000 (Running)"
echo ""
echo "🌐 Opening frontend in your default browser..."
echo ""

# Open the frontend URL in the default browser
if command -v open &> /dev/null; then
    # macOS
    open "http://localhost:8000/frontend.html"
    echo "📱 Frontend opened in browser!"
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open "http://localhost:8000/frontend.html"
    echo "📱 Frontend opened in browser!"
else
    echo "📋 Please manually open this link in your browser:"
    echo "   http://localhost:8000/frontend.html"
fi

echo ""
echo "🎯 Ready to analyze face images!"
echo "======================================="
