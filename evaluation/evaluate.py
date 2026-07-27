import json
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from generate_answer import get_answer_data
from hybrid_query import load_retrieval_backend

def run_evaluation():
    print("Loading Ground Truth Dataset...")
    dataset_path = os.path.join(os.path.dirname(__file__), "ground_truth.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
        
    print(f"Loaded {len(ground_truth)} test cases.\n")
    
    print("Initializing retrieval backend...")
    backend = load_retrieval_backend()
    
    passed = 0
    failed = 0
    failed_details = []
    
    print("Starting Evaluation...\n" + "="*50)
    
    for case in ground_truth:
        q_id = case["id"]
        question = case["question"]
        expected = case["expected_values"]
        
        print(f"[{q_id}/{len(ground_truth)}] Q: {question}")
        data = get_answer_data(backend, question, top_k=5)
        answer = data["answer"]
        
        # Check if all expected strings are in the answer
        success = all(str(val) in answer for val in expected)
        
        if success:
            print("  ✅ PASS")
            passed += 1
        else:
            print("  ❌ FAIL (Hallucination or Miss)")
            failed += 1
            failed_details.append({
                "question": question,
                "expected": expected,
                "actual_answer": answer
            })
            
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    
    total = passed + failed
    accuracy = (passed / total) * 100
    print(f"Total Questions: {total}")
    print(f"Passed: {passed}")
    print(f"Failed (Hallucinations/Misses): {failed}")
    print(f"\nAccuracy: {accuracy:.1f}%\n")
    
    if failed > 0:
        print("Failure Details:")
        for fd in failed_details:
            print(f"  Q: {fd['question']}")
            print(f"  Expected values to be present: {fd['expected']}")
            print(f"  Actual Answer from LLM:\n    {fd['actual_answer']}\n")

if __name__ == "__main__":
    run_evaluation()
