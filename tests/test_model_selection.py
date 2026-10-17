"""Model routing: YAML config loading, task-based selection and fallback chains."""

import yaml

from core.config import ModelConfig, ModelSelector
from tests.conftest import PROJECT_ROOT


def make_selector():
    return ModelSelector(ModelConfig.load_from_yaml(str(PROJECT_ROOT / "model_config.yaml")))


def test_yaml_values_override_defaults():
    with open(PROJECT_ROOT / "model_config.yaml") as f:
        expected = yaml.safe_load(f)["models"]
    config = ModelConfig.load_from_yaml(str(PROJECT_ROOT / "model_config.yaml"))
    for field, value in expected.items():
        assert getattr(config, field) == value


def test_missing_yaml_falls_back_to_defaults(tmp_path):
    config = ModelConfig.load_from_yaml(str(tmp_path / "absent.yaml"))
    assert config == ModelConfig()


def test_seed_task_uses_simple_model_until_complexity_four():
    selector = make_selector()
    assert selector.select_model_for_task("seed", complexity=3) == selector.config.simple_reasoning
    assert selector.select_model_for_task("seed", complexity=4) == selector.config.complex_reasoning


def test_research_task_routes_by_complexity():
    selector = make_selector()
    assert selector.select_model_for_task("research", complexity=2) == selector.config.simple_reasoning
    assert selector.select_model_for_task("research", complexity=3) == selector.config.alternative_llm
    assert selector.select_model_for_task("research", complexity=5) == selector.config.complex_reasoning


def test_sourcing_and_reranking_tasks():
    selector = make_selector()
    assert selector.select_model_for_task("sourcing_live") == selector.config.live_research
    assert selector.select_model_for_task("sourcing_local") == selector.config.embeddings
    assert selector.select_model_for_task("reranking") == selector.config.reranker
    assert selector.select_model_for_task("unknown") == selector.config.simple_reasoning


def test_large_context_overrides_task_type():
    selector = make_selector()
    assert selector.select_model_for_task("seed", context_size=100_001) == selector.config.high_context_fallback
    assert selector.select_model_for_task("seed", context_size=100_000) == selector.config.simple_reasoning


def test_fallback_chains_never_include_the_primary_model():
    config = ModelConfig(
        complex_reasoning="big", simple_reasoning="small", alternative_llm="alt",
        high_context_fallback="long", live_research="sonar",
    )
    selector = ModelSelector(config)
    assert selector.get_fallback_chain("big") == ["alt", "long"]
    assert selector.get_fallback_chain("small") == ["big", "alt"]
    assert selector.get_fallback_chain("sonar") == ["sonar-reasoning-pro", "big"]
    assert selector.get_fallback_chain("something-else") == ["small"]


def test_record_performance_tracks_success_rate_and_latency():
    selector = make_selector()
    selector.record_performance("m", "seed", latency=1.0, success=True)
    selector.record_performance("m", "seed", latency=3.0, success=False)
    metrics = selector.performance_metrics["m_seed"]
    assert metrics["total_requests"] == 2
    assert metrics["success_rate"] == 0.5
    assert metrics["avg_latency"] == 2.0
