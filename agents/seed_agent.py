"""Seed Agent for query decomposition and search planning."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage

from core.config import settings, model_selector
import time

logger = logging.getLogger(__name__)


@dataclass
class SubQuery:
    """A sub-query with priority and search strategy."""
    query: str
    priority: int  # 1-5, where 1 is highest priority
    search_strategy: str  # "local", "live", "both"
    reasoning: str
    

@dataclass
class SearchPlan:
    """Complete search plan from Seed Agent."""
    original_query: str
    sub_queries: List[SubQuery]
    strategy: str  # "local_first", "live_first", "parallel"
    estimated_complexity: int  # 1-5 scale


class SeedAgent:
    """Agent responsible for query decomposition and search planning."""
    
    def __init__(self):
        """Initialize the Seed Agent."""
        # Model will be selected dynamically based on task complexity
        self.openai_api_key = settings.openai_api_key
        
        self.decomposition_prompt = PromptTemplate(
            input_variables=["query"],
            template="""You are the Seed Agent in a multi-agent research system. Your job is to decompose complex research questions into searchable sub-queries and create an optimal search strategy.

Original Query: {query}

Instructions:
1. Break down the query into 2-5 focused sub-queries
2. Assign priority (1=highest, 5=lowest) to each sub-query
3. Choose search strategy for each: "local" (knowledge base), "live" (current web), or "both"
4. Provide reasoning for each decomposition decision
5. Suggest overall search strategy: "local_first", "live_first", or "parallel"

Return your response in this JSON format:
{{
    "sub_queries": [
        {{
            "query": "specific searchable question",
            "priority": 1,
            "search_strategy": "both",
            "reasoning": "why this sub-query is important and why this strategy"
        }}
    ],
    "overall_strategy": "local_first",
    "complexity": 3,
    "reasoning": "overall approach reasoning"
}}

Focus on creating precise, searchable sub-queries that will retrieve the most relevant information."""
        )
    
    def decompose_query(self, query: str) -> SearchPlan:
        """
        Decompose a user query into structured search plan.
        
        Args:
            query: The original user question
            
        Returns:
            SearchPlan with sub-queries and strategy
        """
        start_time = time.time()
        success = False
        
        try:
            # Analyze query complexity to select appropriate model
            complexity_analysis = self.analyze_query_complexity(query)
            complexity = complexity_analysis["complexity_score"]
            
            # Select model based on task complexity
            selected_model = model_selector.select_model_for_task("seed", complexity=complexity)
            
            # Store for error handling
            self.current_model = selected_model
            
            # Create LLM instance with selected model
            llm = self._create_llm_instance(selected_model)
            
            # Generate decomposition using LLM
            prompt = self.decomposition_prompt.format(query=query)
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Parse the LLM response
            plan = self._parse_decomposition_response(query, response.content)
            
            success = True
            logger.info(f"Seed Agent decomposed query into {len(plan.sub_queries)} sub-queries using {selected_model}")
            return plan
            
        except Exception as e:
            logger.error(f"Error in Seed Agent decomposition with {selected_model}: {e}")
            # Try fallback chain
            fallback_models = model_selector.get_fallback_chain(selected_model)
            
            for fallback_model in fallback_models:
                try:
                    logger.info(f"Trying fallback model: {fallback_model}")
                    fallback_llm = self._create_llm_instance(fallback_model)
                    prompt = self.decomposition_prompt.format(query=query)
                    response = fallback_llm.invoke([HumanMessage(content=prompt)])
                    plan = self._parse_decomposition_response(query, response.content)
                    success = True
                    logger.info(f"Fallback successful with {fallback_model}")
                    return plan
                except Exception as fallback_error:
                    logger.warning(f"Fallback model {fallback_model} also failed: {fallback_error}")
                    continue
            
            # Final fallback to simple plan
            return self._create_fallback_plan(query)
            
        finally:
            # Record performance metrics
            execution_time = time.time() - start_time
            model_used = getattr(self, 'current_model', 'unknown')
            model_selector.record_performance(model_used, "seed", execution_time, success)
    
    def _create_llm_instance(self, model_name: str) -> ChatOpenAI:
        """Create LLM instance with specified model."""
        return ChatOpenAI(
            model=model_name,
            openai_api_key=self.openai_api_key,
            temperature=0.1,
            max_tokens=1500
        )
    
    def _parse_decomposition_response(self, original_query: str, response: str) -> SearchPlan:
        """Parse LLM response into SearchPlan object."""
        import json
        
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            
            data = json.loads(json_str)
            
            sub_queries = []
            for sq in data.get("sub_queries", []):
                sub_queries.append(SubQuery(
                    query=sq["query"],
                    priority=sq["priority"], 
                    search_strategy=sq["search_strategy"],
                    reasoning=sq["reasoning"]
                ))
            
            return SearchPlan(
                original_query=original_query,
                sub_queries=sub_queries,
                strategy=data.get("overall_strategy", "local_first"),
                estimated_complexity=data.get("complexity", 3)
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}, using fallback")
            return self._create_fallback_plan(original_query)
    
    def _create_fallback_plan(self, query: str) -> SearchPlan:
        """Create a simple fallback plan when decomposition fails."""
        return SearchPlan(
            original_query=query,
            sub_queries=[SubQuery(
                query=query,
                priority=1,
                search_strategy="both",
                reasoning="Fallback: using original query directly"
            )],
            strategy="local_first",
            estimated_complexity=2
        )
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze query complexity for optimization decisions.
        
        Returns:
            Dictionary with complexity metrics
        """
        # Simple heuristics for query complexity
        word_count = len(query.split())
        has_multiple_concepts = any(word in query.lower() for word in ['and', 'or', 'vs', 'versus', 'compare', 'difference'])
        has_temporal_aspect = any(word in query.lower() for word in ['recent', 'latest', 'current', 'new', 'trend', 'update'])
        
        complexity_score = 1
        if word_count > 10:
            complexity_score += 1
        if has_multiple_concepts:
            complexity_score += 1
        if has_temporal_aspect:
            complexity_score += 1
            
        return {
            "word_count": word_count,
            "complexity_score": min(complexity_score, 5),
            "has_multiple_concepts": has_multiple_concepts,
            "has_temporal_aspect": has_temporal_aspect,
            "recommended_strategy": "live_first" if has_temporal_aspect else "local_first"
        }


# Global instance for easy access
_seed_agent_instance = None

def get_seed_agent() -> SeedAgent:
    """Get global Seed Agent instance."""
    global _seed_agent_instance
    if _seed_agent_instance is None:
        _seed_agent_instance = SeedAgent()
    return _seed_agent_instance