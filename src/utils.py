from pathlib import Path

import open_clip
import torch

# Project paths, resolved relative to this file so scripts work from any cwd
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "RSICD_images"
TXTCLASSES_DIR = DATA_DIR / "txtclasses_rsicd"
JSON_PATH = DATA_DIR / "dataset_rsicd.json"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

classes = [
    "Airport",
    "BareLand",
    "BaseballField",
    "Beach",
    "Bridge",
    "Center",
    "Church",
    "Commercial",
    "DenseResidential",
    "Desert",
    "Farmland",
    "Forest",
    "Industrial",
    "Meadow",
    "MediumResidential",
    "Mountain",
    "Park",
    "Parking",
    "playfields",
    "Playground",
    "Pond",
    "Port",
    "RailwayStation",
    "Resort",
    "River",
    "School",
    "SparseResidential",
    "Square",
    "Stadium",
    "StorageTanks",
    "Viaduct",
]


def load_model_checkpoint(
    model: open_clip.model.CLIP,
    checkpoint_file: str | Path,
    map_location: str | torch.device = "cpu",
) -> open_clip.model.CLIP:
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
