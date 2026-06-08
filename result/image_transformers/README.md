# Image Transformer Graph Suite

Animals-only naturalistic layerwise graphs for ViT and CLIP.

Each subdirectory contains:
- baseline layerwise `(gen_accuracy, ident_accuracy)` graph
- one DAM graph per energy order `n`
- raw and aggregated CSVs
- `config.txt` with exact evaluation parameters
- `insights.md` with concise findings

Settings:
- `easy_s40`
- `easy_s80`
- `occ50_s40`
- `occ50_s80`
- `occ50_dnoise001_s40`
- `occ50_dnoise001_s80`
