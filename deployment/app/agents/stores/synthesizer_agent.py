import sys
from typing import List, AsyncGenerator, Dict, Any, Tuple

from loguru import logger
from pathlib import Path

# LangChain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Local/App Imports
from app.agents.stores.base_agent import BaseAgentNode, AgentState
from app.agents.workflow.initalize import llm_instance, agent_config

class SynthesizerAgent(BaseAgentNode):
    """
    An expert agent responsible for synthesizing information from multiple,
    disparate sources into a single, coherent final answer.

    This agent is only invoked for complex, multi-step plans where the outputs
    of several specialist agents need to be intelligently combined.
    """
    def __init__(self, llm: BaseChatModel, history_k: int = 5):
        agent_name = "SynthesizerAgent"
        # Use the new, focused prompt
        system_prompt = agent_config['synthesizer_agent_prompt']['description']
        
        super().__init__(agent_name=agent_name)
        self.llm = llm
        self.system_prompt = system_prompt
        self.history_k = history_k
        logger.info(f"'{self.agent_name}' initialized for multi-source synthesis.")

    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Executes and returns the final state after the synthesis stream has completed.
        """
        final_state = state
        async for partial_state in self.astream_execute(state):
            final_state = partial_state
        return final_state

    async def astream_execute(self, state: AgentState) -> AsyncGenerator[AgentState, None]:
        """
        Executes the synthesis logic and streams the final, combined answer.
        """
        state = self._prepare_execution(state)
        
        try:
            # The query to answer is the one rewritten by the Triage agent
            query = state.get("rewritten_query") or state.get("original_query", "")
            
            # --- 1. Aggregate all gathered information ---
            # The contexts are the most critical input for this agent.
            # `agent_thinks` is less important here, as we have the final tool outputs.
            all_contexts = state.get("contexts", {})
            logger.info(f"Context pass to SynthesizerAgent: {list(all_contexts.keys())}")
            if not all_contexts:
                logger.warning(f"'{self.agent_name}' was called but no contexts were found. Cannot synthesize.")
                state['agent_response'] = "I apologize, but I was unable to gather the necessary information to answer your multi-part question. Please try rephrasing your request."
                state['is_final_answer'] = True
                yield state
                return

            # Format the contexts for the prompt
            context_str = "\n\n".join(
                [f"--- Information from Source {i+1} ---\n{content}" for i, content in enumerate(all_contexts.values())]
            )
            
            logger.info(f"Synthesizing response for query: '{query}' from {len(all_contexts)} sources.")
            logger.debug(f"Full context for synthesis:\n{context_str}")

            history_messages = self._format_chat_history(state.get("chat_history", []))
            
            # --- 2. Build the Synthesis Prompt ---
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human",
                 "Please synthesize the following information to answer my original question.\n\n"
                 "**My Original Question:**\n{user_query}\n\n"
                 "**Information Gathered by Specialist Agents:**\n{information_gathered}\n\n"
                 "**Your Synthesized Answer:**"
                )
            ])

            chain = prompt | self.llm
            logger.info(f"Streaming synthesized response from '{self.agent_name}'...")

            # --- 3. Execute and Stream the Result ---
            full_response = ""
            async for chunk in chain.astream({
                "chat_history": history_messages,
                "user_query": query,
                "information_gathered": context_str
            }):
                if hasattr(chunk, 'content'):
                    full_response += chunk.content
                    state["agent_response"] = full_response
                    yield state  # Yield updated state with each new chunk

            logger.info(f"Finished streaming synthesized response.")
            state['is_final_answer'] = True
            yield state # Yield the final state

        except Exception as e:
            state = self._handle_execution_error(e, state)
            logger.error(f"Error during synthesis execution: {e}", exc_info=True)
            state['is_final_answer'] = True
            yield state

    def _format_chat_history(self, history: List[Dict[str, str]]) -> List[BaseMessage]:
        """Converts chat history to LangChain message objects."""
        messages = []
        if not history:
            return messages
        
        for item in history[-self.history_k * 2:]:
            role = item.get('role')
            content = item.get('content')
            if role == 'user':
                messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                messages.append(AIMessage(content=content))
        return messages

async def test_synthesizer_agent():
        logger.remove()
        logger.add(sys.stdout, level="INFO")

        synthesizer = SynthesizerAgent(llm=llm_instance)

        print("--- Testing SynthesizerAgent ---")
        # This state simulates the output of the PlanExecutorNode
        # after it has run both a DrugAgent and a CompanyAgent.
        test_state = AgentState(
            rewritten_query="Compare the side effects of Aspirin with the company's return policy for it.",
            contexts={
                "DrugAgent_result": "Aspirin is an NSAID. Common side effects include stomach irritation, heartburn, and increased risk of bleeding. It should not be taken by people with bleeding disorders.",
                "CompanyAgent_result": "Our return policy allows for returns of unopened products within 30 days of purchase with a valid receipt. Prescription medications are non-returnable by law once they have left the pharmacy."
            },
            chat_history=[]
        )
        
        full_answer = ""
        print("Streaming synthesized answer:")
        async for partial_state in synthesizer.astream_execute(test_state):
            new_part = partial_state.get("agent_response", "").replace(full_answer, "", 1)
            print(new_part, end="", flush=True)
            full_answer = partial_state.get("agent_response", "")

        print("\n\n--- Test Complete ---")
        print(f"Full Synthesized Answer:\n{full_answer}")
if __name__ == '__main__':
    import asyncio
    
    

    asyncio.run(test_synthesizer_agent())