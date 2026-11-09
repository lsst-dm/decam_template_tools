import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# The scripts import `common` and the metric modules from verify/calibs itself.
CALIBS_DIR = Path(__file__).resolve().parent.parent
if str(CALIBS_DIR) not in sys.path:
    sys.path.insert(0, str(CALIBS_DIR))
