# scripts/download_models.py
import urllib.request
from pathlib import Path

ZENODO_BASE = "https://zenodo.org/records/<RECORD_ID>/files"

MODELS = {
    "0.2_AuAu/entropy_pt/best_model.keras":   "best_model_AuAu_entropy_pt.keras",
    "0.2_AuAu/entropy_all/best_model.keras":  "best_model_AuAu_entropy_all.keras",
    "0.2_AuAu/energy_pt/best_model.keras":    "best_model_AuAu_energy_pt.keras",
    "0.2_AuAu/energy_all/best_model.keras":   "best_model_AuAu_energy_all.keras",
    "2.76_PbPb/entropy_pt/best_model.keras":  "best_model_PbPb_entropy_pt.keras",
    "2.76_PbPb/entropy_all/best_model.keras": "best_model_PbPb_entropy_all.keras",
    "2.76_PbPb/energy_pt/best_model.keras":   "best_model_PbPb_energy_pt.keras",
    "2.76_PbPb/energy_all/best_model.keras":  "best_model_PbPb_energy_all.keras",
}

root = Path(__file__).resolve().parent.parent / "src" / "trained_models"

for local, remote in MODELS.items():
    dest = root / local
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest}")
        continue
    url = f"{ZENODO_BASE}/{remote}?download=1"
    print(f"[get ] {url}  ->  {dest}")
    urllib.request.urlretrieve(url, dest)