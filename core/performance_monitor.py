"""Performance monitoring system for the AI research pipeline."""

import logging
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance measurement."""
    timestamp: str
    component: str  # "seed_agent", "sourcing_agent", "research_agent", "rag_pipeline"
    operation: str  # "decompose_query", "gather_sources", "synthesize_research", etc.
    model_used: str
    execution_time: float  # seconds
    success: bool
    input_size: int  # tokens or characters
    output_size: int
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    total_requests: int
    successful_requests: int
    average_latency: float
    total_execution_time: float
    success_rate: float
    most_used_model: str
    fastest_model: str
    slowest_model: str
    error_rate: float


class PerformanceMonitor:
    """Monitor and track performance across the pipeline."""
    
    def __init__(self, log_file: str = "performance_metrics.jsonl"):
        """Initialize performance monitor."""
        self.log_file = Path(log_file)
        self.metrics: List[PerformanceMetric] = []
        self.session_start = datetime.now()
        
        # In-memory aggregation for fast access
        self.aggregated_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Load existing metrics if file exists
        self._load_existing_metrics()
    
    def record_operation(self,
                        component: str,
                        operation: str,
                        model_used: str,
                        execution_time: float,
                        success: bool,
                        input_size: int = 0,
                        output_size: int = 0,
                        error_message: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        """
        Record a performance metric for an operation.
        
        Args:
            component: Component name (e.g., "seed_agent")
            operation: Operation name (e.g., "decompose_query")
            model_used: Model identifier used
            execution_time: Time taken in seconds
            success: Whether operation succeeded
            input_size: Input size in tokens/characters
            output_size: Output size in tokens/characters
            error_message: Error message if failed
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            component=component,
            operation=operation,
            model_used=model_used,
            execution_time=execution_time,
            success=success,
            input_size=input_size,
            output_size=output_size,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        # Add to in-memory storage
        self.metrics.append(metric)
        
        # Update aggregated metrics
        self._update_aggregated_metrics(metric)
        
        # Persist to file
        self._persist_metric(metric)
        
        logger.debug(f"Recorded metric: {component}.{operation} - "
                    f"{execution_time:.3f}s - {'✓' if success else '✗'}")
    
    def get_system_metrics(self, 
                          time_window: Optional[timedelta] = None) -> SystemMetrics:
        """
        Get aggregated system metrics.
        
        Args:
            time_window: Time window to consider (None for all time)
            
        Returns:
            SystemMetrics with aggregated data
        """
        # Filter metrics by time window
        filtered_metrics = self.metrics
        if time_window:
            cutoff_time = datetime.now() - time_window
            filtered_metrics = [
                m for m in self.metrics 
                if datetime.fromisoformat(m.timestamp) >= cutoff_time
            ]
        
        if not filtered_metrics:
            return SystemMetrics(
                total_requests=0,
                successful_requests=0,
                average_latency=0.0,
                total_execution_time=0.0,
                success_rate=0.0,
                most_used_model="N/A",
                fastest_model="N/A",
                slowest_model="N/A",
                error_rate=0.0
            )
        
        # Calculate aggregated metrics
        total_requests = len(filtered_metrics)
        successful_requests = sum(1 for m in filtered_metrics if m.success)
        total_execution_time = sum(m.execution_time for m in filtered_metrics)
        average_latency = total_execution_time / total_requests
        success_rate = successful_requests / total_requests
        error_rate = 1.0 - success_rate
        
        # Model usage analysis
        model_usage = {}
        model_performance = {}
        
        for metric in filtered_metrics:
            model = metric.model_used
            
            # Usage counting
            model_usage[model] = model_usage.get(model, 0) + 1
            
            # Performance tracking
            if model not in model_performance:
                model_performance[model] = []
            model_performance[model].append(metric.execution_time)
        
        # Find most used model
        most_used_model = max(model_usage.items(), key=lambda x: x[1])[0] if model_usage else "N/A"
        
        # Find fastest/slowest models
        fastest_model = "N/A"
        slowest_model = "N/A"
        
        if model_performance:
            avg_times = {
                model: sum(times) / len(times) 
                for model, times in model_performance.items()
            }
            fastest_model = min(avg_times.items(), key=lambda x: x[1])[0]
            slowest_model = max(avg_times.items(), key=lambda x: x[1])[0]
        
        return SystemMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            average_latency=average_latency,
            total_execution_time=total_execution_time,
            success_rate=success_rate,
            most_used_model=most_used_model,
            fastest_model=fastest_model,
            slowest_model=slowest_model,
            error_rate=error_rate
        )
    
    def get_component_metrics(self, component: str) -> Dict[str, Any]:
        """Get metrics for a specific component."""
        component_metrics = [m for m in self.metrics if m.component == component]
        
        if not component_metrics:
            return {"error": f"No metrics found for component: {component}"}
        
        # Aggregate by operation
        operations = {}
        for metric in component_metrics:
            op = metric.operation
            if op not in operations:
                operations[op] = {
                    "count": 0,
                    "success_count": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "success_rate": 0.0,
                    "models_used": set()
                }
            
            ops = operations[op]
            ops["count"] += 1
            ops["total_time"] += metric.execution_time
            ops["models_used"].add(metric.model_used)
            
            if metric.success:
                ops["success_count"] += 1
        
        # Calculate averages
        for op_data in operations.values():
            op_data["avg_time"] = op_data["total_time"] / op_data["count"]
            op_data["success_rate"] = op_data["success_count"] / op_data["count"]
            op_data["models_used"] = list(op_data["models_used"])
        
        return {
            "component": component,
            "total_operations": len(component_metrics),
            "operations": operations,
            "overall_success_rate": sum(1 for m in component_metrics if m.success) / len(component_metrics)
        }
    
    def get_model_performance_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Compare performance across different models."""
        model_stats = {}
        
        for metric in self.metrics:
            model = metric.model_used
            if model not in model_stats:
                model_stats[model] = {
                    "usage_count": 0,
                    "success_count": 0,
                    "total_time": 0.0,
                    "execution_times": [],
                    "components": set(),
                    "operations": set()
                }
            
            stats = model_stats[model]
            stats["usage_count"] += 1
            stats["total_time"] += metric.execution_time
            stats["execution_times"].append(metric.execution_time)
            stats["components"].add(metric.component)
            stats["operations"].add(metric.operation)
            
            if metric.success:
                stats["success_count"] += 1
        
        # Calculate derived metrics
        for model, stats in model_stats.items():
            stats["avg_execution_time"] = stats["total_time"] / stats["usage_count"]
            stats["success_rate"] = stats["success_count"] / stats["usage_count"]
            stats["min_time"] = min(stats["execution_times"])
            stats["max_time"] = max(stats["execution_times"])
            stats["components"] = list(stats["components"])
            stats["operations"] = list(stats["operations"])
            
            # Remove raw execution times to keep output clean
            del stats["execution_times"]
        
        return model_stats
    
    def export_metrics(self, 
                      output_file: str,
                      format: str = "json",
                      time_window: Optional[timedelta] = None) -> bool:
        """
        Export metrics to file.
        
        Args:
            output_file: Output file path
            format: Export format ("json", "csv")
            time_window: Time window to export (None for all)
            
        Returns:
            True if export successful
        """
        try:
            # Filter metrics by time window
            metrics_to_export = self.metrics
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics_to_export = [
                    m for m in self.metrics 
                    if datetime.fromisoformat(m.timestamp) >= cutoff_time
                ]
            
            output_path = Path(output_file)
            
            if format.lower() == "json":
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "session_start": self.session_start.isoformat(),
                    "total_metrics": len(metrics_to_export),
                    "system_metrics": asdict(self.get_system_metrics(time_window)),
                    "detailed_metrics": [asdict(m) for m in metrics_to_export]
                }
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
            elif format.lower() == "csv":
                import csv
                
                with open(output_path, 'w', newline='') as f:
                    if metrics_to_export:
                        writer = csv.DictWriter(f, fieldnames=asdict(metrics_to_export[0]).keys())
                        writer.writeheader()
                        for metric in metrics_to_export:
                            writer.writerow(asdict(metric))
            
            logger.info(f"Exported {len(metrics_to_export)} metrics to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return False
    
    def _load_existing_metrics(self):
        """Load existing metrics from file."""
        if not self.log_file.exists():
            return
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        metric_data = json.loads(line)
                        metric = PerformanceMetric(**metric_data)
                        self.metrics.append(metric)
                        self._update_aggregated_metrics(metric)
            
            logger.info(f"Loaded {len(self.metrics)} existing metrics")
            
        except Exception as e:
            logger.warning(f"Error loading existing metrics: {e}")
    
    def _persist_metric(self, metric: PerformanceMetric):
        """Persist a single metric to file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(asdict(metric)) + '\n')
        except Exception as e:
            logger.error(f"Error persisting metric: {e}")
    
    def _update_aggregated_metrics(self, metric: PerformanceMetric):
        """Update in-memory aggregated metrics."""
        key = f"{metric.component}.{metric.operation}.{metric.model_used}"
        
        if key not in self.aggregated_metrics:
            self.aggregated_metrics[key] = {
                "count": 0,
                "success_count": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0
            }
        
        agg = self.aggregated_metrics[key]
        agg["count"] += 1
        agg["total_time"] += metric.execution_time
        agg["min_time"] = min(agg["min_time"], metric.execution_time)
        agg["max_time"] = max(agg["max_time"], metric.execution_time)
        
        if metric.success:
            agg["success_count"] += 1


# Global performance monitor instance
_performance_monitor_instance: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor_instance
    if _performance_monitor_instance is None:
        _performance_monitor_instance = PerformanceMonitor()
    return _performance_monitor_instance


def record_operation(component: str, operation: str, model_used: str, 
                    execution_time: float, success: bool, **kwargs):
    """Convenience function to record an operation."""
    monitor = get_performance_monitor()
    monitor.record_operation(component, operation, model_used, execution_time, success, **kwargs)