"""Command line interface: ``python -m aquatwin <command>``."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from aquatwin.config import AquaTwinConfig


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aquatwin", description="AquaTwin AI digital twin")
    ap.add_argument("--config", default=None, help="YAML config (default: built-in defaults)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate", help="generate the synthetic dataset")
    sub.add_parser("train", help="train detection, localisation and failure models")
    sub.add_parser("calibrate", help="re-calibrate the leak alarm (pipe baselines + false-alarm budget)")
    ev = sub.add_parser("evaluate", help="evaluate under realism levels 0-5")
    ev.add_argument("--episodes", type=int, default=3)
    al = sub.add_parser("all", help="generate + train + evaluate")
    al.add_argument("--episodes", type=int, default=3)
    api = sub.add_parser("api", help="run the REST API")
    api.add_argument("--port", type=int, default=8000)
    dash = sub.add_parser("dashboard", help="run the Streamlit dashboard")
    dash.add_argument("--port", type=int, default=8501)
    sub.add_parser("scenarios", help="list the scenario catalogue")
    args = ap.parse_args(argv)
    cfg = AquaTwinConfig.load(args.config)

    if args.cmd in ("generate", "all"):
        from aquatwin.data.generator import generate_dataset
        m = generate_dataset(cfg)
        print(json.dumps({k: v for k, v in m.items() if k != "config"}, indent=2, default=str))
    if args.cmd in ("train", "all"):
        from aquatwin.pipeline import train_all
        train_all(cfg)
    if args.cmd == "calibrate":
        from aquatwin.pipeline import calibrate_detector
        print(calibrate_detector(cfg))
    if args.cmd in ("evaluate", "all"):
        from aquatwin.evaluation.framework import evaluate_levels
        evaluate_levels(cfg, getattr(args, "episodes", 3))
    if args.cmd == "api":
        import uvicorn
        uvicorn.run("aquatwin.api.main:app", host="0.0.0.0", port=args.port)
    if args.cmd == "dashboard":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", str(args.port)])
    if args.cmd == "scenarios":
        from aquatwin.scenarios.catalog import SCENARIOS
        for s in SCENARIOS.values():
            print(f"{s.category:15s} {s.name:30s} {s.title}")
            for r in s.rules:
                print(f"{'':47s}- {r}")


if __name__ == "__main__":
    main()
