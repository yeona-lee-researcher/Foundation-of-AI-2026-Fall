# Report source (LaTeX)

`main.tex` is the source of `../HW2_Revised_Submission.pdf` (9 pages). Figures are in `figures/` and are referenced by relative paths.

Build with XeLaTeX from this directory (run twice so references settle):

```bash
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
# or: latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Fonts: Carlito (text) and Latin Modern Math (math). With MiKTeX they install on demand (or `miktex packages install carlito lm-math`); with TeX Live they are in the `carlito` and `lm-math` packages. If Carlito is missing the text falls back to TeX Gyre Heros.

Figures `3a_*` to `3d_*` are produced by `../../code/problem3_rks_mlp.py` and `3e_robustness.png` by `../../code/problem3_robustness.py` (regenerated under `outputs/problem3/problem3_outputs/` by `python run_all.py --problem 3` and `--problem robustness`). `P1_*` and `P2_*` are produced by `../../code/report_figures.py` (regenerated under `outputs/report_figures/` by `python run_all.py --problem figures`): `P1_gaussian_probability_explained.png` is an illustrative sketch, `P2_quantity_heavy_tail.png` is drawn from `data/retail.csv`, and the other two from the tables in `results/`.
