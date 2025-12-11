from app.agents.stores.entry_agent import EntryAgentOutput, EntryAgent
from typing import List
from app.agents.workflow.initalize import llm_instance
from app.agents.stores.base_agent import AgentState
from app.scripts.helper import evaluate_entry_agent_results
import asyncio
import json
from pathlib import Path
import time


async def test_entry_agent():
    entry_agent = EntryAgent(llm=llm_instance)
    test_cases  = json.loads(Path("app/tests/data/entry_test.json").read_text())
    predicted_results = []
    for case in test_cases:
        print(f"Testing query: {case['query']}")
        print(f"Expected output: {case['expected_output']}")
        start_time = time.time()
        state = AgentState(
            original_query=case['query'],
            chat_history=[],
            user_role="guest",
        )
        result_state = await entry_agent.aexecute(state)
        print(f"Result State: {result_state}")
        end_time = time.time()
        print(f"Test completed in {end_time - start_time:.2f} seconds")
        predicted_results.append({
            "query": case['query'],
            "intents": result_state.get('intents', [])[0],
            "classified_agent": result_state.get('classified_agent', ""),
            "needs_rewrite": result_state.get('needs_rewrite', False),
            "rewritten_query": result_state.get('rewritten_query', "")
        })
    # Evaluate results
    evaluation_result = evaluate_entry_agent_results(test_cases, predicted_results)
    print("\n--- Evaluation Results ---")
    print(f"Total test cases: {evaluation_result['total_cases']}")
    print(f"Accuracy of intents: {evaluation_result['accuracy_intents']}%")
    print(f"Accuracy of classified agent: {evaluation_result['accuracy_agent']}%")
    print(f"Accuracy of needs rewrite: {evaluation_result['accuracy_rewrite']}%")
    print(f"Overall accuracy: {evaluation_result['accuracy_overall']}%")
    if evaluation_result['mismatches']:
        print("\n--- Mismatches Found ---")
        for mismatch in evaluation_result['mismatches']:
            print(f"Query: {mismatch['query']}")
            print(f"Expected: {mismatch['expected']}")
            print(f"Predicted: {mismatch['predicted']}\n")
if __name__ == "__main__":
    async def main():
            # --- Setup ---
            # EntryAgent không cần tools, chỉ cần LLM
            entry_agent = EntryAgent(llm=llm_instance)
            
            # --- Test Case 1: Simple routing ---
            print("--- Test Case 1: Simple routing to DrugAgent ---")
            state_drug = AgentState(
                original_query="Liều dùng của Paracetamol là gì?",
                chat_history=[],
                user_role="guest",
            )

            result_state_1 = await entry_agent.execute(state_drug)

            print(f"Original Query: {result_state_1.get('original_query')}")
            print(f"Classified Agent: {result_state_1.get('classified_agent')}")
            print(f"Intents: {result_state_1.get('intents')}")
            print(f"Needs Rewrite: {result_state_1.get('needs_rewrite')}")
            print(f"Rewritten Query: {result_state_1.get('rewritten_query')}")
            
            # --- Test Case 2: Query needing a rewrite ---
            print("\n--- Test Case 2: Query needing a rewrite ---")
            state_rewrite = AgentState(
                original_query="thông tin về thuốc đó",
                chat_history=[("Hỏi về Aspirin", "Aspirin là một loại thuốc giảm đau...")],
                user_role="guest",
            )
            
            result_state_2 = await entry_agent.aexecute(state_rewrite)

            print(f"Original Query: {result_state_2.get('original_query')}")
            print(f"Classified Agent: {result_state_2.get('classified_agent')}")
            print(f"Intents: {result_state_2.get('intents')}")
            print(f"Needs Rewrite: {result_state_2.get('needs_rewrite')}")
            print(f"Rewritten Query: {result_state_2.get('rewritten_query')}")

        # Chạy kịch bản test
    # asyncio.run(main())