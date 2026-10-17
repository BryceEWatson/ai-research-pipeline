"""Configuration management for AI Research Pipeline."""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import time
import logging

from dotenv import load_dotenv
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

MODEL_CONFIG_PATH = Path(__file__).resolve().parent.parent / "model_config.yaml"


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    openai_api_key: str = Field(...)
    perplexity_api_key: Optional[str] = Field(None)
    vectara_api_key: Optional[str] = Field(None)
    langsmith_api_key: Optional[str] = Field(None)
    
    # Core Configuration
    chroma_db_path: str = Field("./data/chroma_db")
    log_level: str = Field("INFO")
    embedding_model: str = Field("text-embedding-3-small")
    llm_model: str = Field("gpt-4o")
    
    # RAG Parameters
    chunk_size: int = Field(1500)
    chunk_overlap: int = Field(150)  # 10% of chunk_size
    retrieval_k: int = Field(15)
    min_eval_score: float = Field(0.35)
    
    # Evaluation Settings
    enable_factual_consistency_score: bool = Field(True)
    min_fcs_threshold: float = Field(0.9)
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


class ModelConfig(BaseModel):
    """Model configuration loaded from YAML."""
    
    # Primary reasoning models
    complex_reasoning: str = "o1-2024-12-17"
    simple_reasoning: str = "o1-mini-2024-12-17"
    high_context_fallback: str = "gpt-4.1"
    
    # Alternative for diversity
    alternative_llm: str = "claude-3-5-sonnet-20241022"
    
    # Research & retrieval
    live_research: str = "sonar-pro"
    embeddings: str = "text-embedding-3-large"
    reranker: str = "rerank-english-v3.0"
    
    @classmethod
    def load_from_yaml(cls, yaml_path: str = str(MODEL_CONFIG_PATH)) -> "ModelConfig":
        """Load model configuration from YAML file."""
        try:
            config_path = Path(yaml_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f)
                    return cls(**data.get('models', {}))
            else:
                logger.warning(f"Model config file {yaml_path} not found, using defaults")
                return cls()
        except Exception as e:
            logger.error(f"Error loading model config: {e}, using defaults")
            return cls()


class ModelSelector:
    """Utility class for context-aware model selection."""
    
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.performance_metrics = {}
    
    def select_model_for_task(self, task_type: str, context_size: int = 0, complexity: int = 1) -> str:
        """
        Select optimal model based on task type and context.
        
        Args:
            task_type: Type of task (seed, sourcing, research, etc.)
            context_size: Estimated context size in tokens
            complexity: Task complexity (1-5 scale)
            
        Returns:
            Selected model identifier
        """
        # High context fallback
        if context_size > 100000:  # 100k tokens
            return self.config.high_context_fallback
        
        # Task-specific model selection
        if task_type == "seed":
            # Simple reasoning for query decomposition
            return self.config.simple_reasoning if complexity <= 3 else self.config.complex_reasoning
        
        elif task_type == "sourcing_live":
            # Live research API
            return self.config.live_research
        
        elif task_type == "sourcing_local":
            # Use current embedding model for local search
            return self.config.embeddings
        
        elif task_type == "research":
            # Complex reasoning for synthesis
            if complexity >= 4:
                return self.config.complex_reasoning
            elif complexity >= 3:
                return self.config.alternative_llm  # Claude for diversity
            else:
                return self.config.simple_reasoning
        
        elif task_type == "reranking":
            return self.config.reranker
        
        else:
            # Default fallback
            return self.config.simple_reasoning
    
    def get_fallback_chain(self, primary_model: str) -> list[str]:
        """Get fallback chain for a primary model."""
        fallback_chains = {
            self.config.complex_reasoning: [self.config.alternative_llm, self.config.high_context_fallback],
            self.config.simple_reasoning: [self.config.complex_reasoning, self.config.alternative_llm],
            self.config.alternative_llm: [self.config.complex_reasoning, self.config.high_context_fallback],
            self.config.live_research: ["sonar-reasoning-pro", self.config.complex_reasoning],
        }
        return fallback_chains.get(primary_model, [self.config.simple_reasoning])
    
    def record_performance(self, model: str, task_type: str, latency: float, success: bool):
        """Record performance metrics for model optimization."""
        key = f"{model}_{task_type}"
        if key not in self.performance_metrics:
            self.performance_metrics[key] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_latency": 0.0,
                "avg_latency": 0.0,
                "success_rate": 0.0
            }
        
        metrics = self.performance_metrics[key]
        metrics["total_requests"] += 1
        metrics["total_latency"] += latency
        metrics["avg_latency"] = metrics["total_latency"] / metrics["total_requests"]
        
        if success:
            metrics["successful_requests"] += 1
        
        metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]


def load_settings() -> Settings:
    """Load settings from environment variables and .env file."""
    # Load .env file if it exists
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
    
    return Settings()


# Global instances
settings = load_settings()
model_config = ModelConfig.load_from_yaml()
model_selector = ModelSelector(model_config)