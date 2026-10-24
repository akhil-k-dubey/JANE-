#!/bin/bash
# ======================================================================
# Launch Project Jane Live Interactive Voice Session
# ======================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "======================================================================"
echo "Starting Project Jane Live Microphone Runtime..."
echo "Speak into your microphone while Jane executes her plan!"
echo "Commands: 'Also check the pressure', 'Vent gas', 'STOP!'"
echo "======================================================================"

python3 scripts/live_interactive_session.py "$@"
