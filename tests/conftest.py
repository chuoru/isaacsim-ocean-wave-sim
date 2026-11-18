import sys
from pathlib import Path

# The pure-numpy core is imported as top-level `wave_sim`: importing the real `isaacsim` package boots Kit.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exts" / "isaacsim.ocean.wave_sim" / "isaacsim" / "ocean"))
