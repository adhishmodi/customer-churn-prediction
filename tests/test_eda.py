def test_eda_module_import_is_side_effect_free():
    from src import eda

    assert callable(eda.run_eda)
    assert eda.FIGURES_DIR.name == "figures"
