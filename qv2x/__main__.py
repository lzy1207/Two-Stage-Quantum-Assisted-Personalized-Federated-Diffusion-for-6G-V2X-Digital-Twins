import argparse
from pathlib import Path

from .pipeline import read_config, run, infer, run_sweep, plot_results


def main():
    parser = argparse.ArgumentParser(description="Quantum federated diffusion REM reconstruction")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "demo", "stage1", "conditions", "stage2", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/demo.json" if name == "demo" else "configs/paper.json")
        command.add_argument("--output", default="runs/demo" if name == "demo" else "runs/paper")
        command.add_argument("--device", choices=["cpu", "cuda", "auto"])
        command.add_argument("--resume", action="store_true", help="Resume stage2 from stage2_last.pt (use with stage2)")
    query = sub.add_parser("infer")
    query.add_argument("--run", required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--output", required=True)
    query.add_argument("--seed", type=int, default=42)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--config", default="configs/paper.json")
    sweep.add_argument("--output", default="runs/sweep")
    sweep.add_argument("--variants", nargs="+", default=["full", "classical", "no_entanglement", "no_prototype"])
    sweep.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    plot = sub.add_parser("plot")
    plot.add_argument("--run", required=True)
    stability = sub.add_parser("stability")
    stability.add_argument("--run", required=True)
    stability.add_argument("--realization-id", type=int)
    stability.add_argument("--samples", type=int, default=100)
    stability.add_argument("--output")
    stability.add_argument("--seed", type=int, default=7000)
    importer = sub.add_parser("import-data")
    importer.add_argument("--manifest", required=True)
    importer.add_argument("--output", required=True)
    importer.add_argument("--png-db-min", type=float)
    importer.add_argument("--png-db-max", type=float)
    importer.add_argument("--value-kind", choices=["received_power", "gain", "pathloss"], default="received_power")
    importer.add_argument("--tx-power-dbm", type=float)
    importer.add_argument("--coordinates", choices=["normalized", "pixel"], default="normalized")
    args = parser.parse_args()
    if args.command == "infer":
        infer(args.run, args.query, args.output, args.seed)
    elif args.command == "sweep":
        run_sweep(read_config(args.config), args.output, args.variants, args.seeds)
    elif args.command == "plot":
        plot_results(args.run)
    elif args.command == "stability":
        from .stability import evaluate_stability
        evaluate_stability(args.run, realization_id=args.realization_id, samples=args.samples,
                           output_dir=args.output, seed=args.seed)
    elif args.command == "import-data":
        from .import_data import import_manifest
        import_manifest(args.manifest, args.output, png_db_min=args.png_db_min, png_db_max=args.png_db_max,
                        value_kind=args.value_kind, tx_power_dbm=args.tx_power_dbm, coordinates=args.coordinates)
    else:
        config = read_config(args.config)
        if args.device:
            config["device"] = args.device
        if args.resume and args.command != "stage2":
            parser.error("--resume must be used with stage2 to avoid retraining Stage I")
        run(config, args.output, stage="all" if args.command in ("run", "demo") else args.command, resume=args.resume)


if __name__ == "__main__":
    main()
