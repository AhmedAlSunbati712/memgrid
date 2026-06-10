from pathlib import Path

from handoff.lib.graph_display import (
    LEGACY_ANIMALS_SETTING_NAMES,
    LEGACY_ANIMALS_SETTINGS,
    list_setting_pngs,
)

HANDOFF_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_animals_settings_catalog():
    assert len(LEGACY_ANIMALS_SETTINGS) == 6
    assert LEGACY_ANIMALS_SETTING_NAMES == (
        "easy_s40",
        "easy_s80",
        "occ50_s40",
        "occ50_s80",
        "occ50_dnoise001_s40",
        "occ50_dnoise001_s80",
    )


def test_list_setting_pngs_sort_order(tmp_path: Path):
    setting_dir = tmp_path / "easy_s40"
    setting_dir.mkdir()
    names = [
        "dam_n8_layers_gen_vs_ident.png",
        "baseline_layers_gen_vs_ident.png",
        "dam_n2_layers_gen_vs_ident.png",
        "dam_n6_layers_gen_vs_ident.png",
        "dam_n4_layers_gen_vs_ident.png",
    ]
    for name in names:
        (setting_dir / name).write_bytes(b"png")
    ordered = [p.name for p in list_setting_pngs(setting_dir)]
    assert ordered == [
        "baseline_layers_gen_vs_ident.png",
        "dam_n2_layers_gen_vs_ident.png",
        "dam_n4_layers_gen_vs_ident.png",
        "dam_n6_layers_gen_vs_ident.png",
        "dam_n8_layers_gen_vs_ident.png",
    ]


def test_layer_artifacts_have_five_pngs_each():
    root = HANDOFF_ROOT / "results" / "image_transformers"
    for name in LEGACY_ANIMALS_SETTING_NAMES:
        pngs = list_setting_pngs(root / name)
        assert len(pngs) == 5, f"{name}: expected 5 PNGs, got {len(pngs)}"


def test_similarity_fixed_artifacts_have_five_pngs_each():
    root = HANDOFF_ROOT / "results" / "image_transformers_similarity_ordered_fixed"
    for name in LEGACY_ANIMALS_SETTING_NAMES:
        pngs = list_setting_pngs(root / name)
        assert len(pngs) == 5, f"{name}: expected 5 PNGs, got {len(pngs)}"
        assert all("ordered_by_distance" in p.name for p in pngs)
