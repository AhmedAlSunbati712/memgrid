from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.run_animals_image_transformer_graphs import DAM_ORDERS, SETTING_SPECS, _shift_label


def test_setting_specs_cover_expected_animals_regimes():
    names = [setting.name for setting in SETTING_SPECS]
    assert names == [
        "easy_s40",
        "easy_s80",
        "occ50_s40",
        "occ50_s80",
        "occ50_dnoise001_s40",
        "occ50_dnoise001_s80",
    ]


def test_dam_orders_match_requested_energy_powers():
    assert DAM_ORDERS == (2, 4, 6, 8)


def test_shift_label_smoke():
    assert _shift_label(2.0, 2.0) == "right/up"
    assert _shift_label(-2.0, 0.0) == "left/flat"
