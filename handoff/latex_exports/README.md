# REPORT PDF (memoir)

Source: [`../REPORT.md`](../REPORT.md)

Output: [`pdf/REPORT.pdf`](pdf/REPORT.pdf) (Pandoc + `memoir` + Tectonic)

## Build

```bash
cd handoff
bash latex_exports/compile_memo.sh
```

Requires [Tectonic](https://tectonic-typesetting.github.io/) at `/Volumes/youwhat/tools/tectonic` (or set `TECTONIC=`). Cache/tmp default to `/Volumes/youwhat/tmp/` so builds work when the system disk is full.

## Figures

`REPORT.md` embeds bundled PNGs with markdown image syntax, e.g. `![](results/grid/multiscale/tradeoff_by_n.png)`.

The compile script symlinks `latex_exports/pdf/results` → `../../results` so Tectonic can resolve `\includegraphics{results/...}` when building inside `pdf/`.

## Layout notes

- `patch_memo_tex.py` widens section 7 script tables and fixes inline math before Tectonic runs.
- Section 7 script tables use three columns (script | purpose & args | output).
