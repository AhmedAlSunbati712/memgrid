from pathlib import Path

from vision.run_naturalistic_dam_hard_ident import HARD_IDENT_BRANCHES


def test_hard_ident_branches_match_animals_and_fruits_vit_cls_layer11():
    assert len(HARD_IDENT_BRANCHES) == 2
    animals = HARD_IDENT_BRANCHES[0]
    fruits = HARD_IDENT_BRANCHES[1]

    assert animals["category"] == "animals"
    assert animals["model_name"] == "vit_base_patch16_224"
    assert animals["pooling"] == "cls"
    assert animals["layer"] == "layer_11"
    assert animals["corruption_mode"] == "occlusion"
    assert animals["occlusion_frac"] == 0.5
    assert animals["decision_noise_levels"] == (0.0, 0.01)

    assert fruits["category"] == "fruits"
    assert fruits["model_name"] == "vit_base_patch16_224"
    assert fruits["pooling"] == "cls"
    assert fruits["layer"] == "layer_11"
    assert fruits["corruption_mode"] == "occlusion"
    assert fruits["occlusion_frac"] == 0.5
