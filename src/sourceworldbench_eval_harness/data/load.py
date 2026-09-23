import logging
from pathlib import Path

import typer

from sourceworldbench_eval_harness.artifacts import write, write_config_beside
from sourceworldbench_eval_harness.config import load_config
from sourceworldbench_eval_harness.hub import load_split, load_token


def main(config: Path) -> None:
    """Pull the split named by the `load` block and save it to disk."""
    load_token()
    cfg = load_config(config)
    if not cfg.load.output:
        raise ValueError("The standalone load command requires a non-empty `load.output` path")

    # This step exists to fetch, so it ignores the cache by default; every other
    # reader of the same block reuses it.
    ds = load_split(
        cfg.load,
        "load",
        download_mode=cfg.load.download_mode or "force_redownload",
    )

    out = Path(cfg.shared.output_dir) / cfg.load.output
    write(out, ds)
    write_config_beside(out, cfg)
    logging.info("[load] %d rows -> %s", len(ds), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    typer.run(main)
