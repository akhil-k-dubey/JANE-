#!/usr/bin/env python3
"""Project Jane Interactive Terminal Chat Entrypoint."""

import sys
import os

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.interactive_chat import JaneTerminalChat

if __name__ == "__main__":
    chat = JaneTerminalChat()
    chat.run()
