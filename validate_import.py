import sys
from unittest.mock import MagicMock
sys.modules['runpod'] = MagicMock()
import handler
print("Successfully imported handler module!")
