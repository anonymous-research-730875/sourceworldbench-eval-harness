"""Push local metrics into a Hugging Face results dataset.

One self-contained file per submission at `submissions/<slug>/<model>__<stamp>.json`
— append-only, no race conditions. The Gradio Space globs them all.
"""

DEFAULT_RESULTS_REPO = "anonymous-research-730875/sourceworldbench-results"
