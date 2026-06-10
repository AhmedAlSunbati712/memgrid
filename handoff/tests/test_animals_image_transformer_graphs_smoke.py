from pathlib import Path

from vision.run_animals_image_transformer_graphs import DAM_ORDERS, SETTING_SPECS, _shift_label


def test_setting_specs_cover_expected_animals_regimes():
    names = [setting.name for setting in SETTING_SPECS]
    assert names == ["hard_s40", "hard_s80"]
    for setting in SETTING_SPECS:
        assert setting.corruption_mode == "noise_shift_occlusion"
        assert setting.decision_noise_std == 0.01
        assert setting.occlusion_frac == 0.5


def test_dam_orders_match_requested_energy_powers():
    assert DAM_ORDERS == (2, 4, 6, 8)


def test_shift_label_smoke():
    assert _shift_label(2.0, 2.0) == "right/up"
    assert _shift_label(-2.0, 0.0) == "left/flat"
