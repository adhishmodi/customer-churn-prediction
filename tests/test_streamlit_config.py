from pathlib import Path
import tomllib


def test_streamlit_config_loads_render_secret_file_and_upload_limit():
    config_path = Path(__file__).parents[1] / ".streamlit" / "config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["server"]["maxUploadSize"] == 10
    assert config["secrets"]["files"] == ["/etc/secrets/secrets.toml"]
