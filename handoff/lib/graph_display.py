from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

LEGACY_ANIMALS_SETTINGS: tuple[dict[str, str], ...] = (
    {
        "name": "easy_s40",
        "title": "easy_s40 — noise+shift, n_stored=40",
        "corruption_summary": "noise_shift, noise_std=5, max_shift=2, decision_noise=0",
    },
    {
        "name": "easy_s80",
        "title": "easy_s80 — noise+shift, n_stored=80",
        "corruption_summary": "noise_shift, noise_std=5, max_shift=2, decision_noise=0",
    },
    {
        "name": "occ50_s40",
        "title": "occ50_s40 — 50% occlusion, n_stored=40",
        "corruption_summary": "occlusion, occlusion_frac=0.5, decision_noise=0",
    },
    {
        "name": "occ50_s80",
        "title": "occ50_s80 — 50% occlusion, n_stored=80",
        "corruption_summary": "occlusion, occlusion_frac=0.5, decision_noise=0",
    },
    {
        "name": "occ50_dnoise001_s40",
        "title": "occ50_dnoise001_s40 — occlusion + decision noise, n_stored=40",
        "corruption_summary": "occlusion, occlusion_frac=0.5, decision_noise=0.01",
    },
    {
        "name": "occ50_dnoise001_s80",
        "title": "occ50_dnoise001_s80 — occlusion + decision noise, n_stored=80",
        "corruption_summary": "occlusion, occlusion_frac=0.5, decision_noise=0.01",
    },
)

LEGACY_ANIMALS_SETTING_NAMES: tuple[str, ...] = tuple(s["name"] for s in LEGACY_ANIMALS_SETTINGS)

_DAM_ORDER = {"baseline": 0, "dam_n2": 1, "dam_n4": 2, "dam_n6": 3, "dam_n8": 4}


def _png_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name.startswith("baseline"):
        return (_DAM_ORDER["baseline"], name)
    match = re.match(r"dam_n(\d+)_", name)
    if match:
        order_key = f"dam_n{match.group(1)}"
        return (_DAM_ORDER.get(order_key, 99), name)
    return (99, name)


def list_setting_pngs(setting_dir: Path) -> list[Path]:
    if not setting_dir.is_dir():
        return []
    return sorted(setting_dir.glob("*.png"), key=_png_sort_key)


def read_setting_config_summary(setting_dir: Path) -> str | None:
    config_path = setting_dir / "config.txt"
    if not config_path.exists():
        return None
    fields: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    if not fields:
        return None
    return (
        f"corruption={fields.get('corruption_mode', '?')}, "
        f"noise_std={fields.get('noise_std', '?')}, "
        f"max_shift={fields.get('max_shift', '?')}, "
        f"occlusion_frac={fields.get('occlusion_frac', '?')}, "
        f"decision_noise={fields.get('decision_noise_std', '?')}"
    )


def display_setting_graphs(
    artifact_root: Path,
    setting_name: str,
    *,
    title: str | None = None,
    display_fn: Callable[[Any], None] | None = None,
    image_cls: type | None = None,
) -> list[Path]:
    setting_dir = artifact_root / setting_name
    if display_fn is None or image_cls is None:
        from IPython.display import Image, display

        display_fn = display
        image_cls = Image

    header = title or setting_name
    print(header)
    if not setting_dir.is_dir():
        print(f"Missing setting folder: {setting_dir}")
        return []

    config_summary = read_setting_config_summary(setting_dir)
    if config_summary:
        print(config_summary)

    pngs = list_setting_pngs(setting_dir)
    if not pngs:
        print(f"No PNG files in {setting_dir}")
        return []

    for png in pngs:
        print(png.name)
        display_fn(image_cls(filename=str(png)))
    return pngs
