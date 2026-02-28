#!/bin/bash
# MindGap — One-Command Launcher
# Starts the Flask API server and opens the browser

set -e

cd "$(dirname "$0")"

echo ""
echo "🧠 MindGap — AI Research Gap Discovery"
echo "======================================="
echo ""

# Activate virtualenv
source venv/bin/activate

# Check dependencies
python -c "import flask, flask_cors" 2>/dev/null || {
  echo "📦 Installing missing dependencies..."
  pip install flask flask-cors -q
}

echo "🚀 Starting Flask API server..."
echo "   Frontend: http://localhost:5050/"
   echo "   API:      http://localhost:5050/api/health"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

# Open browser after 3-second delay (model loading takes a moment)
(sleep 3 && open http://localhost:5050/) &

python server.py
