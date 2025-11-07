"""
Complete RAG pipeline demo script.

Demonstrates: Load data -> Build index -> Retrieve examples -> Build prompt 
-> Call LLM (offline) -> Validate output

Run with: python demo.py
"""

import os
import json
from retriever import load_examples, build_index, get_top_k
from prompt_builder import build_prompt
from llm_client import call_llm, parse_llm_text_to_json
from validate import validate_output, coverage_metric
from utils import load_json, pretty_json, log


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    """Run the complete RAG pipeline demo."""
    print("\n" + "█" * 80)
    print("  WORKFLOW ASSISTANT RAG - COMPLETE PIPELINE DEMO")
    print("█" * 80)
    
    # Load schema and examples
    print_section("STEP 1: Load Schema and Examples")
    
    schema_path = os.path.join("data", "schema.json")
    examples_path = os.path.join("data", "examples.json")
    
    try:
        schema = load_json(schema_path)
        log(f"✓ Loaded schema from {schema_path}")
        print(f"Schema preview:\n{pretty_json(schema)}\n")
    except FileNotFoundError:
        log(f"✗ Schema not found, using fallback schema")
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string"},
                "condition": {"type": "object"},
                "action": {"type": "object"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]}
            },
            "required": ["id", "type", "action"]
        }
    
    try:
        examples = load_examples(examples_path)
        log(f"✓ Loaded {len(examples)} examples from {examples_path}")
    except FileNotFoundError:
        log(f"✗ Examples not found, using fallback examples")
        examples = [
            {
                "id": "ex-001",
                "title": "Email on delay",
                "description": "Send email notification when task duration exceeds threshold",
                "config": {
                    "id": "notify-delay",
                    "type": "notification",
                    "condition": {"field": "duration", "operator": ">", "value": 120},
                    "action": {"type": "email", "recipients": ["team@example.com"]},
                    "priority": "high"
                }
            }
        ]
    
    # Build retrieval index
    print_section("STEP 2: Build Retrieval Index")
    
    log("Building TF-IDF index from examples...")
    vectorizer, tfidf_matrix, texts = build_index(examples)
    log(f"✓ Index built with vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Number of indexed examples: {len(examples)}")
    
    # Query and retrieve relevant examples
    print_section("STEP 3: Retrieve Relevant Examples")
    
    user_query = "send notification email when a task takes longer than 2 hours"
    
    log(f"User query: '{user_query}'")
    
    k = 3
    retrieved = get_top_k(user_query, vectorizer, tfidf_matrix, examples, k=k)
    
    log(f"✓ Retrieved top {len(retrieved)} examples")
    print("\nRetrieved examples:")
    for i, (example, score) in enumerate(retrieved, 1):
        print(f"  {i}. ID: {example['id']} | Score: {score} | Title: {example['title']}")
    
    # Build LLM prompt
    print_section("STEP 4: Build LLM Prompt")
    
    log("Composing prompt with schema, examples, and user query...")
    prompt = build_prompt(user_query, retrieved, schema)
    
    log(f"✓ Prompt built ({len(prompt)} characters)")
    print("\nPrompt preview (first 500 chars):")
    print("-" * 80)
    print(prompt[:500] + "...")
    print("-" * 80)
    print(f"\nFull prompt length: {len(prompt)} characters")
    
    # Call LLM (offline mode)
    print_section("STEP 5: Call LLM")
    
    log("Calling LLM client...")
    llm_result = call_llm(prompt, model="gpt-4")
    
    if llm_result.get("offline"):
        log("⚠ Running in OFFLINE mode (no API key set)")
        print(f"Note: {llm_result.get('note')}")
        print("\nThe prompt shown above would be sent to the LLM.")
        
        # Simulated LLM response for demo
        log("\n✓ Simulating LLM response for demo purposes...")
        
        fake_response_json = {
            "id": "wf-email-duration",
            "type": "notification",
            "condition": {
                "field": "duration",
                "operator": ">",
                "value": 7200,
                "unit": "seconds"
            },
            "action": {
                "type": "email",
                "recipients": ["admin@example.com", "team@example.com"],
                "subject": "Task Duration Alert",
                "body": "A task has exceeded 2 hours duration"
            },
            "priority": "high"
        }
        
        fake_response = json.dumps(fake_response_json, indent=2)
        
        print("\n📝 Simulated LLM Response:")
        print("-" * 80)
        print(fake_response)
        print("-" * 80)
        
    elif llm_result.get("error"):
        log(f"✗ LLM call failed: {llm_result.get('message')}")
        fake_response = "{}"
    else:
        log("✓ LLM response received")
        fake_response = llm_result.get("text", "{}")
        print(f"\nLLM Response:\n{fake_response}")
    
    # Parse and validate LLM output
    print_section("STEP 6: Parse and Validate Output")
    
    log("Parsing LLM response...")
    parsed_output, parse_error = parse_llm_text_to_json(fake_response)
    
    if parse_error:
        log(f"✗ Failed to parse LLM output: {parse_error}")
        print("\nValidation cannot proceed without valid JSON.")
        return
    
    log("✓ Successfully parsed JSON from LLM response")
    print(f"\nParsed output:\n{pretty_json(parsed_output)}")
    
    print("\n" + "-" * 80)
    log("Validating output against schema...")
    
    is_valid, errors = validate_output(parsed_output, schema)
    coverage = coverage_metric(parsed_output, schema)
    
    print("\n📊 Validation Results:")
    print("-" * 80)
    
    if is_valid:
        print("✅ Status: VALID")
        print("   The generated workflow config conforms to the schema.")
    else:
        print("❌ Status: INVALID")
        print("\n   Errors:")
        for error in errors:
            print(f"   - {error}")
    
    print(f"\n📈 Coverage Metric: {coverage:.1%}")
    print(f"   ({int(coverage * 100)}% of required fields present)")
    
    # Summary
    print_section("DEMO SUMMARY")
    
    print(f"""
Pipeline Stages Completed:
  ✓ Loaded schema and {len(examples)} examples
  ✓ Built TF-IDF retrieval index
  ✓ Retrieved {len(retrieved)} relevant examples for query
  ✓ Composed {len(prompt)}-character prompt
  ✓ Called LLM (offline mode: returned placeholder)
  ✓ Simulated LLM response with valid JSON
  ✓ Validated output: {'VALID' if is_valid else 'INVALID'} ({coverage:.1%} coverage)

Query: "{user_query}"

Retrieved Example IDs: {', '.join([ex['id'] for ex, _ in retrieved])}

Output Status: {'✅ Valid workflow configuration' if is_valid else '❌ Invalid or incomplete'}

Next Steps:
  1. Add real examples to data/examples.json
  2. Define your schema in data/schema.json
  3. Set OPENAI_API_KEY in .env to enable real LLM calls
  4. Run the Flask app with: python app.py
""")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
