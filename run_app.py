"""One-command launcher: fetch the data if it is missing, then start the Streamlit app.

    python run_app.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
needed = ["Indian_pines_corrected.mat", "Salinas_corrected.mat", "Cuprite_f970619t01p02_r02_sc03.a.rfl.mat",
          "92AV3C.spc", "splib07a"]
if not all((ROOT / "data" / "raw" / n).exists() for n in needed):
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "download_data.py")])
sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                          "--server.headless", "true"] + sys.argv[1:]))
