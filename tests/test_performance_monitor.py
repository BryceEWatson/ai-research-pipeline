"""Performance metrics: recording, persistence to JSONL and aggregation."""

import json
from types import SimpleNamespace

import core.performance_monitor as performance_monitor
from agents.seed_agent import SeedAgent
from core.performance_monitor import PerformanceMonitor


def record(monitor, model, seconds, success=True, component="seed_agent"):
    monitor.record_operation(
        component=component, operation="decompose_query", model_used=model,
        execution_time=seconds, success=success,
    )


def test_metrics_persist_and_reload(tmp_path):
    log = tmp_path / "metrics.jsonl"
    record(PerformanceMonitor(str(log)), "small", 0.5)
    assert len(log.read_text().splitlines()) == 1
    assert len(PerformanceMonitor(str(log)).metrics) == 1


def test_system_metrics_aggregate(tmp_path):
    monitor = PerformanceMonitor(str(tmp_path / "m.jsonl"))
    record(monitor, "small", 1.0)
    record(monitor, "small", 3.0, success=False)
    record(monitor, "big", 5.0)
    metrics = monitor.get_system_metrics()
    assert metrics.total_requests == 3
    assert metrics.successful_requests == 2
    assert metrics.most_used_model == "small"
    assert metrics.average_latency == 3.0


def test_model_comparison_and_export(tmp_path):
    monitor = PerformanceMonitor(str(tmp_path / "m.jsonl"))
    record(monitor, "small", 1.0)
    comparison = monitor.get_model_performance_comparison()
    assert comparison["small"]["usage_count"] == 1
    out = tmp_path / "export.json"
    assert monitor.export_metrics(str(out), format="json")
    assert json.loads(out.read_text())


def test_seed_agent_calls_are_logged(tmp_path, monkeypatch):
    monitor = PerformanceMonitor(str(tmp_path / "m.jsonl"))
    monkeypatch.setattr(performance_monitor, "_performance_monitor_instance", monitor)
    fake_llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="not json"))
    agent = SeedAgent()
    monkeypatch.setattr(agent, "_create_llm_instance", lambda model_name: fake_llm)

    agent.decompose_query("What is RAG?")

    assert [m.component for m in monitor.metrics] == ["seed_agent"]
    assert monitor.metrics[0].model_used == agent.current_model
    assert (tmp_path / "m.jsonl").exists()


def test_fallback_success_is_logged_under_the_fallback_model(tmp_path, monkeypatch):
    monitor = PerformanceMonitor(str(tmp_path / "m.jsonl"))
    monkeypatch.setattr(performance_monitor, "_performance_monitor_instance", monitor)
    calls = []

    def make_llm(model_name):
        def invoke(messages):
            calls.append(model_name)
            if len(calls) == 1:
                raise RuntimeError("primary model down")
            return SimpleNamespace(content="not json")
        return SimpleNamespace(invoke=invoke)

    agent = SeedAgent()
    monkeypatch.setattr(agent, "_create_llm_instance", make_llm)

    agent.decompose_query("What is RAG?")

    assert len(calls) == 2
    assert monitor.metrics[-1].model_used == calls[1]
    assert monitor.metrics[-1].success is True


def test_research_with_no_sources_logs_nothing(tmp_path, monkeypatch):
    from agents.research_agent import ResearchAgent

    monitor = PerformanceMonitor(str(tmp_path / "m.jsonl"))
    monkeypatch.setattr(performance_monitor, "_performance_monitor_instance", monitor)

    result = ResearchAgent().synthesize_research("What is RAG?", SimpleNamespace(documents=[]))

    assert result.total_sources == 0
    assert monitor.metrics == []
