"""Build structured prompts for LLM workflow generation."""

import json
from typing import Any


def pretty_examples_for_prompt(retrieved: list[tuple[dict[str, Any], float]]) -> str:
    """Format retrieved examples for the prompt."""
    if not retrieved:
        return "No examples available.\n"
    
    examples_text = []
    for i, (example, score) in enumerate(retrieved, 1):
        example_id = example.get('id', f'unknown-{i}')
        description = example.get('description', 'No description')
        config = example.get('config', {})
        
        example_block = f"Example {i} (ID: {example_id}, Relevance: {score}):\n"
        example_block += f"Description: {description}\n"
        example_block += f"Config:\n{json.dumps(config, indent=2)}\n"
        
        examples_text.append(example_block)
    
    return "\n".join(examples_text)


def build_prompt(
    user_query: str,
    retrieved: list[tuple[dict[str, Any], float]],
    schema: dict[str, Any]
) -> str:
    """Build complete LLM prompt with schema, examples, and user query."""
    
    system_instruction = (
        "You are an assistant that returns a single JSON object which must strictly "
        "conform to the provided JSON schema. Do not return any text explanation or commentary."
    )
    
    schema_section = "JSON Schema:\n"
    schema_section += json.dumps(schema, indent=2)
    schema_section += "\n"
    
    if 'required' in schema:
        required_fields = schema['required']
        schema_section += f"\nRequired fields: {', '.join(required_fields)}\n"
    
    examples_section = "Few-shot Examples:\n"
    examples_section += "=" * 80 + "\n"
    examples_section += pretty_examples_for_prompt(retrieved)
    
    user_request_section = f"User Request:\n{user_query}\n"
    
    final_instructions = (
        "\nInstructions:\n"
        "- Return only a single JSON object that conforms to the schema above.\n"
        "- Return valid JSON only.\n"
        "- Do not wrap the JSON in code fences.\n"
        "- Do not include any explanatory text before or after the JSON.\n"
        "- If you cannot comply, return an empty JSON object {}."
    )
    
    prompt = f"{system_instruction}\n\n"
    prompt += f"{schema_section}\n"
    prompt += f"{examples_section}\n"
    prompt += f"{user_request_section}\n"
    prompt += f"{final_instructions}\n"
    
    return prompt


if __name__ == "__main__":
    sample_schema = {
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
    
    sample_retrieved = [
        (
            {
                "id": "ex-001",
                "title": "Email notification on delay",
                "description": "Send email when task duration exceeds threshold",
                "config": {
                    "id": "notify-delay",
                    "type": "notification",
                    "condition": {"field": "duration", "operator": ">", "value": 120},
                    "action": {"type": "email", "recipients": ["team@example.com"]},
                    "priority": "high"
                }
            },
            0.8542
        ),
        (
            {
                "id": "ex-002",
                "title": "Slack alert for overdue tasks",
                "description": "Post to Slack when task is overdue",
                "config": {
                    "id": "slack-overdue",
                    "type": "notification",
                    "condition": {"field": "status", "operator": "==", "value": "overdue"},
                    "action": {"type": "slack", "channel": "#alerts", "message": "Task is overdue"},
                    "priority": "medium"
                }
            },
            0.7231
        )
    ]
    
    test_query = "send email when task late"
    
    print("Building prompt for query:", test_query)
    print("=" * 80)
    print()
    
    prompt = build_prompt(test_query, sample_retrieved, sample_schema)
    
    print(prompt)
    print("=" * 80)
    print(f"\nPrompt length: {len(prompt)} characters")
