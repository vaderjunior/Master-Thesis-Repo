import sys
from pathlib import Path

# put the project root on sys.path so tests can import `src.hsrag...`
sys.path.insert(0, str(Path(__file__).parent))