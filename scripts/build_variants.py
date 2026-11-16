"""Dev script: build all variants for one registered model. Replaced by the CLI in Phase 5."""

import sys
from pathlib import Path

from edgeopt.data import load_text_dataset, load_vision_dataset
from edgeopt.logging_config import configure_logging
from edgeopt.models import load_model
from edgeopt.pipeline import build_variants


def main(name: str) -> None:
    configure_logging("INFO")
    spec, model = load_model(name)
    if spec.task == "vision":
        calib = load_vision_dataset("train", Path("data"), limit=100)
    else:
        calib = load_text_dataset("train", limit=128)
    built = build_variants(spec, model, calib.feeds, Path("artifacts"))
    for v in built.values():
        print(f"{v.name:<13} {v.size_mb:8.1f} MB  {v.node_count:5d} nodes  {v.path}")


if __name__ == "__main__":
    main(sys.argv[1])
