"""Dev script: benchmark every built variant of one model. Replaced by the CLI in Phase 5."""

import sys
from pathlib import Path

from edgeopt.benchmark import benchmark_variants, discover_variants, render_table
from edgeopt.data import load_text_dataset, load_vision_dataset
from edgeopt.logging_config import configure_logging
from edgeopt.models import get_spec


def main(name: str) -> None:
    configure_logging("INFO")
    spec = get_spec(name)
    if spec.task == "vision":
        dataset = load_vision_dataset("val", Path("data"))
    else:
        dataset = load_text_dataset("validation")
    model_dir = Path("artifacts") / name
    report = benchmark_variants(name, discover_variants(model_dir), dataset)
    report.write_json(model_dir / "benchmark.json")
    print(render_table(report))


if __name__ == "__main__":
    main(sys.argv[1])