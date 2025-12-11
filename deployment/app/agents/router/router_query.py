from app.agents.router.improved_route_tool import ImprovedRouteTool, MAPPING_ROTER_TO_AGENT
from app.agents.router.router_intent import RouterIntent

import os
from loguru import logger
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.agents.workflow.state import GraphState



class RouterQuery:
    name: str = "RouterQuery"
    description: str = "RouterQuery is responsible for routing queries to the appropriate agent based on the intent of the query. It uses a combination of improved routing techniques and intent recognition to determine the best agent to handle the query."
    def __init__(self):
        self.router_agent = ImprovedRouteTool()
        self.router_intent = RouterIntent(model_path="app/agents/router/pretrained_router.json")



    def route_query(self, state: GraphState) -> Dict[str, Any]:
        """
        Routes the query to the appropriate agent based on the intent of the query.
        
        Args:
            query (str): The query to be routed.
        
        Returns:
            Dict[str, Any]: A dictionary containing the routing result.
        """
        query = state.get("original_query", "")
        logger.info(f"Routing query: {query}")
        
        
        # Determine the intent of the query
        intent = self.router_intent.route_query(query)
        logger.info(f"Identified intent: {intent.name}")
        logger.info(f"Similarity score: {intent.similarity_score}")

        # Route the query based on the identified intent
        route_results = self.router_agent.route_query(query)
        logger.info(f"Identified Agent: {route_results.name}")
        logger.info(f"Similarity score: {route_results.similarity_score}")
        agent = MAPPING_ROTER_TO_AGENT.get(intent, "NaiveAgent")

        logger.info(f"Routing result: {'query': query, 'intents': intent, 'classified_agent': agent, 'route_results': route_results}")
        
        partial_state = {
            "query": query,
            "intents": intent.name,
            "classified_agent": agent,
        }
        
        return {**state, **partial_state}