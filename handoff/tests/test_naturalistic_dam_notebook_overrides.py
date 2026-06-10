from handoff.lib.vision_runner import naturalistic_dam_notebook_overrides


def test_notebook_quick_overrides():
    quick = naturalistic_dam_notebook_overrides(quick=True)
    assert quick["category"] == "animals"
    assert quick["max_stage_a_configs"] == 1
    assert quick["skip_stage_b"] is True
    assert quick["include_clip"] is False


def test_notebook_full_overrides_are_capped():
    full = naturalistic_dam_notebook_overrides(quick=False)
    assert "category" not in full
    assert full["max_stage_a_configs"] == 6
    assert full["skip_stage_b"] is True
    assert full["n_seeds"] == 1
    assert full["storage_sizes"] == [40]
    assert full["include_clip"] is True
