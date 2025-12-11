def evaluate_entry_agent_results(test_cases, predictions):
    """
    Đánh giá kết quả đầu ra của agent EntryAnalyzer.
    
    Args:
        test_cases (list): Danh sách dict chứa 'query' và 'expected_output'
        predictions (list): Danh sách dict chứa kết quả thực tế trả về từ agent

    Returns:
        dict: Thống kê độ chính xác và các trường sai khác
    """
    assert len(test_cases) == len(predictions), "Số lượng kết quả không khớp với số test case"

    correct_intent = 0
    correct_agent = 0
    correct_rewrite = 0
    correct_all = 0
    total = len(test_cases)

    mismatches = []

    for i, (case, pred) in enumerate(zip(test_cases, predictions), start=1):
        expected = case["expected_output"]
        pred_norm = {
            "intents": str(pred.get("intents", "")).strip().lower(),
            "classified_agent": str(pred.get("classified_agent", "")).strip(),
            "needs_rewrite": bool(pred.get("needs_rewrite", False))
        }

        match_intent = pred_norm["intents"] == expected["intents"]
        match_agent = pred_norm["classified_agent"] == expected["classified_agent"]
        match_rewrite = pred_norm["needs_rewrite"] == expected["needs_rewrite"]

        if match_intent: correct_intent += 1
        if match_agent: correct_agent += 1
        if match_rewrite: correct_rewrite += 1
        if match_intent and match_agent and match_rewrite:
            correct_all += 1
        else:
            mismatches.append({
                "query": case["query"],
                "expected": expected,
                "predicted": pred_norm
            })

    result = {
        "total_cases": total,
        "accuracy_intents": round(correct_intent / total * 100, 2),
        "accuracy_agent": round(correct_agent / total * 100, 2),
        "accuracy_rewrite": round(correct_rewrite / total * 100, 2),
        "accuracy_overall": round(correct_all / total * 100, 2),
        "mismatches": mismatches
    }


    return result
