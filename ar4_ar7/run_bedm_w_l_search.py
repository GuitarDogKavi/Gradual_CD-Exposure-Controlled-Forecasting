import subprocess
import sys
import time
from pathlib import Path

python_exe = sys.executable

BASE_DIR = Path(r"C:\Users\mpkhd\Desktop\Final_Year_Project_Code\ar4_ar7")
sim_dirs = ["sim1", "sim2", "sim3"]

for sim in sim_dirs:
    script_path = BASE_DIR / sim / "bedm_w_l_search" / "w_l_search_metrics.py"

    print(f"\n=== Running {script_path} ===\n")

    start = time.time()

    subprocess.run(
        [python_exe, str(script_path)],
        check=True
    )

    elapsed = time.time() - start
    print(f"Finished {sim} in {elapsed:.1f} seconds")