#!/bin/bash
# MindGap Explorer - Quick Demo Script

echo "🚀 MindGap Explorer - AI-Powered Research Gap Discovery"
echo "========================================================"
echo ""
echo "📊 Regenerating visualization with latest AI predictions..."
echo ""

cd "$(dirname "$0")"
source venv/bin/activate
python scripts/visualize_credible_ai.py

echo ""
echo "✅ Opening visualization in browser..."
open data/graph_credible_ai.html

echo ""
echo "🎯 Demonstration Features:"
echo "   • Top-Right: Model transparency (99.72% ROC-AUC)"
echo "   • Bottom-Right: Top 20 AI predictions"
echo "   • Bottom-Left: Methodology explanation"
echo ""
echo "🎮 Controls:"
echo "   • Rotate: Click & drag"
echo "   • Zoom: Scroll or +/- buttons"
echo "   • Info: Hover over nodes"
echo ""
echo "✨ Ready for presentation!"
