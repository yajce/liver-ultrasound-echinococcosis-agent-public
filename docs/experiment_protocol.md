# Experiment Protocol

1. The fixed `dev` split is used for Skill development, error review, and version iteration.
2. The fixed `test` split is a held-out test set.
3. Test results must not be used to add, remove, or tune Skill rules before the final Skill version is selected.
4. The model, benchmark images and labels, baseline prompt, parser, and evaluation logic remain fixed across comparisons.
5. The Skill document is the primary experimental variable.
6. Every Skill result records both the declared Skill version and the SHA256 hash of the exact Skill file content.
7. Every model and Skill version uses the same split files under `data/splits/`.

The split is generated from `data/labels.csv` by `scripts/create_split.py` with random seed 42. Skill comparisons default to `dev`; use `test` only for the explicitly held-out evaluation after selecting the final Skill version.
