"""CLI commands that work without network access."""

from click.testing import CliRunner

import core.performance_monitor as performance_monitor
from app.cli import cli
from core.config import model_config


def test_documented_commands_are_registered():
    expected = {"ask", "add", "add-file", "stats", "report", "reset", "models", "performance", "select-model"}
    assert expected <= set(cli.commands)


def test_models_command():
    result = CliRunner().invoke(cli, ["models"])
    assert result.exit_code == 0, result.output
    assert model_config.simple_reasoning in result.output
    assert model_config.live_research in result.output


def test_select_model_command():
    result = CliRunner().invoke(cli, ["select-model", "research", "--complexity", "4"])
    assert result.exit_code == 0, result.output
    assert model_config.complex_reasoning in result.output


def test_performance_command_accepts_its_options(tmp_path, monkeypatch):
    monitor = performance_monitor.PerformanceMonitor(str(tmp_path / "m.jsonl"))
    monitor.record_operation("seed_agent", "decompose_query", "small", 0.2, True)
    monkeypatch.setattr(performance_monitor, "_performance_monitor_instance", monitor)
    out = tmp_path / "export.json"
    result = CliRunner().invoke(cli, ["performance", "--component", "seed_agent", "--export", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
