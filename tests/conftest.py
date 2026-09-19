"""Cho phép `import src...` khi chạy pytest từ thư mục gốc repo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
