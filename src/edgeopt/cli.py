"""Command-line interface: edgeopt {list-models, build, benchmark, run-all}."""

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import get_args

from edgeopt.benchmark import (
    BASELINE,
    DEFAULT_RUNS,
    DEFAULT_WARMUP,
    benchmark_variants,
    discover_variants,
    render_table,
)
from edgeopt.config import Settings
from edgeopt.data import load_calibration, load_eval
from edgeopt.exceptions import ConfigError, EdgeOptError
from edgeopt.logging_config import configure_logging
from edgeopt.models import REGISTRY, get_spec, load_model
from edgeopt.pipeline import ALL_VARIANTS, DEFAULT_VARIANTS, BuiltVariant, build_variants
from edgeopt.quantize import Calibration

CALIBRATIONS: tuple[str, ...] = get_args(Calibration)


def select_variants(
    available: Mapping[str, Path], requested: Sequence[str] | None
) -> dict[str, Path]:
    """Pick the variants to benchmark. The fp32 baseline is always included."""
    if requested is None:
        return dict(available)
    wanted = {BASELINE, *requested}
    missing = sorted(wanted - set(available))
    if missing:
        raise ConfigError(f"Variants not built: {missing}. Available: {sorted(available)}")
    return {name: available[name] for name in sorted(wanted)}


def _add_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", choices=sorted(REGISTRY), help="Registered model name")


def _add_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--variants", nargs="+", choices=ALL_VARIANTS, metavar="VARIANT",
        help=f"Variants to build (fp32 is always built). Choices: {', '.join(ALL_VARIANTS)}",
    )  # fmt: skip
    parser.add_argument("--calibration", choices=CALIBRATIONS, default="minmax")
    parser.add_argument("--calib-samples", type=int, default=None, help="Calibration set size")
    parser.add_argument(
        "--op-types", nargs="+", metavar="OP", default=None,
        help="Restrict static INT8 to these ONNX op types, e.g. MatMul",
    )  # fmt: skip


def _add_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=None, help="Random eval subset size")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None, help="JSON report path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edgeopt",
        description="Convert, quantize and benchmark PyTorch models with ONNX Runtime.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-models", help="Show supported models")

    build = sub.add_parser("build", help="Export, optimize and quantize a model")
    _add_model_arg(build)
    _add_build_args(build)

    bench = sub.add_parser("benchmark", help="Benchmark already-built variants")
    _add_model_arg(bench)
    bench.add_argument(
        "--variants", nargs="+", metavar="VARIANT", default=None,
        help="Only these variants (fp32 is always included)",
    )  # fmt: skip
    _add_bench_args(bench)

    run_all = sub.add_parser("run-all", help="Build, then benchmark what was built")
    _add_model_arg(run_all)
    _add_build_args(run_all)
    _add_bench_args(run_all)
    return parser


def _list_models() -> int:
    for name, spec in sorted(REGISTRY.items()):
        print(f"{name:<18}{spec.task:<8}{spec.source}")
    return 0


def _run_build(args: argparse.Namespace, settings: Settings) -> dict[str, BuiltVariant]:
    spec, model = load_model(args.model)
    variants = tuple(args.variants) if args.variants else DEFAULT_VARIANTS[spec.task]
    # Only fetch calibration data when static quantization will actually use it.
    calib_feeds = (
        load_calibration(spec, settings.data_dir, args.calib_samples).feeds
        if "int8_static" in variants
        else []
    )
    built = build_variants(
        spec,
        model,
        calib_feeds,
        settings.artifact_dir,
        variants=variants,
        calibration=args.calibration,
        op_types=args.op_types,
    )
    for v in built.values():
        print(f"{v.name:<13}{v.size_mb:>9.1f} MB{v.node_count:>7} nodes  {v.path}")
    return built


def _run_benchmark(args: argparse.Namespace, paths: Mapping[str, Path], settings: Settings) -> int:
    dataset = load_eval(get_spec(args.model), settings.data_dir, args.limit)
    report = benchmark_variants(args.model, paths, dataset, args.warmup, args.runs, args.threads)
    out = args.output or settings.artifact_dir / args.model / "benchmark.json"
    report.write_json(out)
    print(render_table(report))
    print(f"Report written to {out}")
    return 0


def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "list-models":
        return _list_models()
    if args.command == "build":
        _run_build(args, settings)
        return 0
    if args.command == "benchmark":
        available = discover_variants(settings.artifact_dir / args.model)
        return _run_benchmark(args, select_variants(available, args.variants), settings)
    built = _run_build(args, settings)  # run-all
    return _run_benchmark(args, {name: v.path for name, v in built.items()}, settings)


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env(env)
        configure_logging(settings.log_level)
        return _dispatch(args, settings)
    except EdgeOptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
