from sarych.env_report import collect_env_report, write_env_report


def test_env_report_collects_and_writes_without_crashing(tmp_path):
    text = collect_env_report()
    assert "Python" in text
    assert "Platform" in text

    path = tmp_path / "env_report.txt"
    write_env_report(path)
    assert path.exists()
    assert "Current working directory" in path.read_text(encoding="utf-8")
