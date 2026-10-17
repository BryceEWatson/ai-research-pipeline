"""Command-line interface for AI Research Pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import settings, model_selector
from core.rag_pipeline import get_rag_pipeline
from core.performance_monitor import get_performance_monitor

# Initialize console for rich output
console = Console()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """AI Research Pipeline - Multi-Agent RAG Assistant."""
    pass


@cli.command()
@click.argument('question', type=str)
@click.option('--k', default=None, type=int, help='Number of documents to retrieve')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def ask(question: str, k: Optional[int], verbose: bool):
    """Ask a research question and get an answer."""
    if verbose:
        console.print(f"[blue]Question:[/blue] {question}")
        console.print(f"[blue]Retrieving documents (k={k or settings.retrieval_k})...[/blue]")
    
    try:
        # Initialize RAG pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            
            # Show progress
            task = progress.add_task("Initializing RAG pipeline...", total=None)
            rag_pipeline = get_rag_pipeline()
            
            progress.update(task, description="Processing question...")
            response = rag_pipeline.answer_question(question, k)
        
        # Display results
        _display_answer(response, verbose)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if verbose:
            import traceback
            console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")
        sys.exit(1)


@cli.command()
@click.argument('text', type=str)
@click.option('--title', default="", help='Document title')
@click.option('--url', default="", help='Source URL')
@click.option('--score', default=0.7, type=float, help='Evaluation score (0-1)')
def add(text: str, title: str, url: str, score: float):
    """Add a document to the knowledge base."""
    console.print(f"[blue]Adding document:[/blue] {title or 'Untitled'}")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            
            task = progress.add_task("Adding document to knowledge base...", total=None)
            rag_pipeline = get_rag_pipeline()
            
            ids = rag_pipeline.add_document(
                text=text,
                title=title,
                source_url=url,
                eval_score=score
            )
            
        console.print(f"[green]✓[/green] Added document with {len(ids)} chunks")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--title', default="", help='Document title (defaults to filename)')
@click.option('--url', default="", help='Source URL')
@click.option('--score', default=0.7, type=float, help='Evaluation score (0-1)')
def add_file(file_path: str, title: str, url: str, score: float):
    """Add a document from file to the knowledge base."""
    path = Path(file_path)
    
    # Use filename as title if not provided
    if not title:
        title = path.stem
    
    console.print(f"[blue]Adding file:[/blue] {path}")
    
    try:
        # Read file content
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            
            task = progress.add_task("Processing file...", total=None)
            rag_pipeline = get_rag_pipeline()
            
            ids = rag_pipeline.add_document(
                text=text,
                title=title,
                source_url=url or f"file://{path.absolute()}",
                eval_score=score
            )
            
        console.print(f"[green]✓[/green] Added file with {len(ids)} chunks")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)


@cli.command()
def stats():
    """Show knowledge base statistics."""
    try:
        rag_pipeline = get_rag_pipeline()
        stats = rag_pipeline.get_knowledge_base_stats()
        
        # Create stats panel
        stats_text = f"""
Total Documents: {stats['total_documents']}
Collection: {stats['collection_name']}
Embedding Model: {stats['embedding_model']}
LLM Model: {stats['llm_model']}
"""
        
        panel = Panel(
            stats_text.strip(),
            title="Knowledge Base Statistics",
            border_style="blue"
        )
        
        console.print(panel)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--detailed', is_flag=True, help='Show detailed document information')
@click.option('--export', type=click.Path(), help='Export report to JSON file')
def report(detailed: bool, export: str):
    """Generate comprehensive knowledge base report."""
    try:
        rag_pipeline = get_rag_pipeline()
        report_data = rag_pipeline.vector_store.get_knowledge_base_report()
        
        if "error" in report_data:
            console.print(f"[red]Error generating report:[/red] {report_data['error']}")
            sys.exit(1)
        
        # Export to JSON if requested
        if export:
            import json
            with open(export, 'w') as f:
                json.dump(report_data, f, indent=2)
            console.print(f"[green]✓[/green] Report exported to {export}")
        
        # Display the report
        _display_knowledge_report(report_data, detailed)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to reset the knowledge base?')
def reset():
    """Reset the knowledge base (delete all documents)."""
    try:
        rag_pipeline = get_rag_pipeline()
        rag_pipeline.vector_store.reset_collection()
        console.print("[green]✓[/green] Knowledge base reset successfully")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)


@cli.command()
def models():
    """Show current model configuration and selection strategy."""
    try:
        console.print("[blue]Current Model Configuration:[/blue]")
        
        # Create model configuration table
        config_table = Table(title="Model Configuration")
        config_table.add_column("Task Type", style="cyan")
        config_table.add_column("Primary Model", style="green")
        config_table.add_column("Purpose", style="yellow")
        
        config_table.add_row("Simple Reasoning", model_selector.config.simple_reasoning, "Query decomposition, planning")
        config_table.add_row("Complex Reasoning", model_selector.config.complex_reasoning, "Synthesis, conflict detection")
        config_table.add_row("High Context", model_selector.config.high_context_fallback, "Large document processing")
        config_table.add_row("Alternative LLM", model_selector.config.alternative_llm, "Diverse perspectives")
        config_table.add_row("Live Research", model_selector.config.live_research, "Real-time web research")
        config_table.add_row("Embeddings", model_selector.config.embeddings, "Vector similarity search")
        config_table.add_row("Reranker", model_selector.config.reranker, "Relevance optimization")
        
        console.print(config_table)
        
        # Show selection strategy info
        strategy_text = """
Model Selection Strategy:
• Seed Agent: Uses simple reasoning (o1-mini) for fast query decomposition
• Research Agent: Uses complex reasoning (o1) for synthesis and conflict detection
• Sourcing Agent: Uses live research model (sonar-pro) for web queries
• High context fallback: Automatically switches for large document sets (>100k tokens)
• Fallback chains: Each model has backup options if primary fails
        """
        
        console.print(Panel(strategy_text.strip(), title="Selection Strategy", border_style="blue"))
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--component', type=str, help='Filter by component (seed_agent, sourcing_agent, research_agent)')
@click.option('--export', type=click.Path(), help='Export metrics to JSON file')
def performance(component: Optional[str], export: Optional[str]):
    """Show model performance metrics and statistics."""
    try:
        monitor = get_performance_monitor()
        
        console.print("[blue]System Performance Metrics:[/blue]")
        
        # Get overall system metrics
        system_metrics = monitor.get_system_metrics()
        
        # Create system metrics table
        sys_table = Table(title="System Overview")
        sys_table.add_column("Metric", style="cyan")
        sys_table.add_column("Value", style="green")
        
        sys_table.add_row("Total Requests", str(system_metrics.total_requests))
        sys_table.add_row("Success Rate", f"{system_metrics.success_rate:.1%}")
        sys_table.add_row("Average Latency", f"{system_metrics.average_latency:.3f}s")
        sys_table.add_row("Total Execution Time", f"{system_metrics.total_execution_time:.2f}s")
        sys_table.add_row("Most Used Model", system_metrics.most_used_model)
        sys_table.add_row("Fastest Model", system_metrics.fastest_model)
        sys_table.add_row("Slowest Model", system_metrics.slowest_model)
        
        console.print(sys_table)
        
        # Model performance comparison
        model_perf = monitor.get_model_performance_comparison()
        if model_perf:
            console.print("\n[blue]Model Performance Comparison:[/blue]")
            
            perf_table = Table()
            perf_table.add_column("Model", style="cyan")
            perf_table.add_column("Usage Count", style="green")
            perf_table.add_column("Success Rate", style="yellow")
            perf_table.add_column("Avg Time", style="red")
            perf_table.add_column("Components Used", style="blue")
            
            for model, stats in model_perf.items():
                perf_table.add_row(
                    model,
                    str(stats["usage_count"]),
                    f"{stats['success_rate']:.1%}",
                    f"{stats['avg_execution_time']:.3f}s",
                    ", ".join(stats["components"][:2]) + ("..." if len(stats["components"]) > 2 else "")
                )
            
            console.print(perf_table)
        
        # Component-specific metrics if requested
        if component:
            console.print(f"\n[blue]Component Metrics: {component}[/blue]")
            comp_metrics = monitor.get_component_metrics(component)
            
            if "error" not in comp_metrics:
                comp_table = Table()
                comp_table.add_column("Operation", style="cyan")
                comp_table.add_column("Count", style="green")
                comp_table.add_column("Success Rate", style="yellow")
                comp_table.add_column("Avg Time", style="red")
                
                for op, data in comp_metrics["operations"].items():
                    comp_table.add_row(
                        op,
                        str(data["count"]),
                        f"{data['success_rate']:.1%}",
                        f"{data['avg_time']:.3f}s"
                    )
                
                console.print(comp_table)
            else:
                console.print(f"[yellow]{comp_metrics['error']}[/yellow]")
        
        # Export if requested
        if export:
            if monitor.export_metrics(export, format="json"):
                console.print(f"[green]✓[/green] Performance metrics exported to {export}")
            else:
                console.print(f"[red]Error exporting metrics to {export}[/red]")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument('task_type', type=click.Choice(['seed', 'sourcing_live', 'sourcing_local', 'research', 'reranking']))
@click.option('--complexity', default=3, type=int, help='Task complexity (1-5)')
@click.option('--context-size', default=0, type=int, help='Estimated context size in tokens')
def select_model(task_type: str, complexity: int, context_size: int):
    """Show which model would be selected for a given task type and complexity."""
    try:
        selected_model = model_selector.select_model_for_task(
            task_type=task_type,
            complexity=complexity,
            context_size=context_size
        )
        
        fallback_chain = model_selector.get_fallback_chain(selected_model)
        
        # Create selection table
        selection_table = Table(title=f"Model Selection for {task_type}")
        selection_table.add_column("Parameter", style="cyan")
        selection_table.add_column("Value", style="green")
        
        selection_table.add_row("Task Type", task_type)
        selection_table.add_row("Complexity", str(complexity))
        selection_table.add_row("Context Size", f"{context_size:,} tokens")
        selection_table.add_row("Selected Model", selected_model)
        selection_table.add_row("Fallback Chain", " → ".join(fallback_chain))
        
        console.print(selection_table)
        
        # Show reasoning
        reasoning_text = f"""
Selection Reasoning:
• Task type '{task_type}' with complexity {complexity}
• Context size: {context_size:,} tokens
• Selected: {selected_model}
• Fallback options: {len(fallback_chain)} models available
        """
        
        console.print(Panel(reasoning_text.strip(), title="Selection Logic", border_style="green"))
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)


def _display_answer(response: dict, verbose: bool = False):
    """Display the answer in a formatted way."""
    # Create answer panel
    answer_panel = Panel(
        response['answer'],
        title="Answer",
        border_style="green"
    )
    console.print(answer_panel)
    
    # Determine phase and show appropriate information
    phase = response.get('phase', '1_basic_rag')
    
    if phase == "2_multi_agent":
        # Phase 2: Multi-agent response
        if verbose:
            _display_phase2_details(response)
    else:
        # Phase 1: Basic RAG response
        if verbose:
            _display_phase1_details(response)


def _display_phase1_details(response: dict):
    """Display Phase 1 (basic RAG) details."""
    # Show sources if available
    if response.get('sources'):
        console.print("\n[blue]Sources used:[/blue]")
        for i, source in enumerate(response['sources'], 1):
            score = source.get('eval_score', 0.0)
            score_color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
            
            console.print(f"  [{i}] {source.get('title', 'Unknown')}")
            console.print(f"      URL: {source.get('url', 'Unknown')}")
            console.print(f"      Score: [{score_color}]{score:.2f}[/{score_color}]")
    
    # Show basic metadata
    context_used = response.get('context_used', 0)
    console.print(f"\n[dim]Retrieved {context_used} documents[/dim]")


def _display_phase2_details(response: dict):
    """Display Phase 2 (multi-agent) details."""
    # Show enhanced metadata
    console.print(f"\n[cyan]Multi-Agent Analysis:[/cyan]")
    console.print(f"  Confidence Score: {response.get('confidence_score', 0):.2f}")
    console.print(f"  Total Sources: {response.get('total_sources', 0)}")
    
    source_breakdown = response.get('source_breakdown', {})
    console.print(f"  Local Sources: {source_breakdown.get('local', 0)}")
    console.print(f"  Live Sources: {source_breakdown.get('live', 0)}")
    
    # Show search strategy
    search_plan = response.get('search_plan', {})
    console.print(f"  Strategy: {search_plan.get('strategy', 'unknown')}")
    console.print(f"  Complexity: {search_plan.get('complexity', 0)}")
    
    # Show quality assessment
    quality_assessment = response.get('quality_assessment', {})
    if quality_assessment:
        console.print(f"\n[magenta]Quality Assessment:[/magenta]")
        overall_confidence = quality_assessment.get('overall_confidence', 0)
        fcs_score = quality_assessment.get('fcs_score', 0)
        
        # Color confidence based on value
        confidence_color = "green" if overall_confidence > 0.8 else "yellow" if overall_confidence > 0.6 else "red"
        console.print(f"  Overall Confidence: [{confidence_color}]{overall_confidence:.3f}[/{confidence_color}]")
        
        if fcs_score > 0:
            fcs_color = "green" if fcs_score > 0.7 else "yellow" if fcs_score > 0.4 else "red"
            console.print(f"  FCS Score: [{fcs_color}]{fcs_score:.3f}[/{fcs_color}]")
        
        explanation = quality_assessment.get('explanation', '')
        if explanation:
            console.print(f"  Assessment: {explanation}")
        
        # Show quality metrics if available
        metrics = quality_assessment.get('metrics', {})
        if metrics:
            console.print(f"  Quality Metrics:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    console.print(f"    {metric}: {value:.3f}")
                else:
                    console.print(f"    {metric}: {value}")
    
    # Show conflicts if any
    conflicts = response.get('conflicts', [])
    if conflicts:
        console.print(f"\n[yellow]Conflicts Detected ({len(conflicts)}):[/yellow]")
        for conflict in conflicts:
            console.print(f"  • {conflict.get('claim', 'Unknown')}")
            console.print(f"    Severity: {conflict.get('severity', 'unknown')}")
    
    # Show citations
    citations = response.get('citations', [])
    if citations:
        console.print(f"\n[blue]Citations ({len(citations)}):[/blue]")
        for citation in citations:
            console.print(f"  [{citation.get('id', 'Unknown')}] {citation.get('title', 'Unknown')}")
            console.print(f"      Type: {citation.get('type', 'unknown')}")
            console.print(f"      URL: {citation.get('url', 'Unknown')}")
    
    # Show execution time
    exec_time = response.get('execution_time', 0)
    console.print(f"\n[dim]Execution time: {exec_time:.2f}s[/dim]")


def _display_knowledge_report(report_data: dict, detailed: bool = False):
    """Display comprehensive knowledge base report."""
    from rich.table import Table
    from rich.tree import Tree
    
    # Overview panel
    overview_content = f"""Total Documents: {report_data['total_documents']}
Unique Sources: {report_data['unique_sources']}
Total Content: {report_data['total_content_length']:,} characters
Average Doc Length: {report_data['average_document_length']:.0f} characters"""
    
    overview_panel = Panel(
        overview_content,
        title="📊 Knowledge Base Overview",
        border_style="green"
    )
    console.print(overview_panel)
    
    # Topics analysis
    topics = report_data.get('topics', [])
    if topics:
        console.print("\n[cyan]🧠 Knowledge Topics:[/cyan]")
        topics_table = Table(show_header=True, header_style="bold magenta")
        topics_table.add_column("Topic", style="dim")
        topics_table.add_column("Mentions", justify="center")
        topics_table.add_column("Relevance", justify="center")
        
        for topic in topics:
            relevance_color = "green" if topic['relevance'] == 'high' else "yellow" if topic['relevance'] == 'medium' else "red"
            topics_table.add_row(
                topic['topic'],
                str(topic['mentions']),
                f"[{relevance_color}]{topic['relevance']}[/{relevance_color}]"
            )
        
        console.print(topics_table)
    
    # Sources summary
    sources = report_data.get('sources', [])
    if sources:
        console.print(f"\n[cyan]📚 Sources ({len(sources)}):[/cyan]")
        for i, source in enumerate(sources[:10], 1):  # Show first 10
            console.print(f"  {i}. {source}")
        if len(sources) > 10:
            console.print(f"  ... and {len(sources) - 10} more sources")
    
    # Document details (if requested)
    if detailed and report_data.get('documents'):
        console.print(f"\n[cyan]📄 Document Details:[/cyan]")
        docs_table = Table(show_header=True, header_style="bold blue")
        docs_table.add_column("Title", style="dim", max_width=30)
        docs_table.add_column("Source", style="dim", max_width=40)
        docs_table.add_column("Score", justify="center")
        docs_table.add_column("Length", justify="center")
        docs_table.add_column("Preview", max_width=50)
        
        for doc in report_data['documents'][:20]:  # Show first 20
            score_color = "green" if doc['eval_score'] > 0.7 else "yellow" if doc['eval_score'] > 0.4 else "red"
            docs_table.add_row(
                doc['title'],
                doc['source_url'],
                f"[{score_color}]{doc['eval_score']:.2f}[/{score_color}]",
                str(doc['content_length']),
                doc['content_preview']
            )
        
        console.print(docs_table)
        
        if len(report_data['documents']) > 20:
            console.print(f"\n[dim]... and {len(report_data['documents']) - 20} more documents[/dim]")
    
    # Summary
    if not detailed:
        console.print(f"\n[dim]💡 Use --detailed flag to see full document information[/dim]")
        console.print(f"[dim]💾 Use --export filename.json to save report[/dim]")


if __name__ == '__main__':
    cli()