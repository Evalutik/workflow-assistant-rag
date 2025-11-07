"""Utility functions for JSON I/O, formatting, and logging."""

import json
import os
from datetime import datetime
from typing import Any


def load_json(path: str) -> Any:
    """Load and parse a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    """Save object to JSON file. Creates parent directories if needed."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def pretty_json(obj: Any, indent: int = 2) -> str:
    """Convert object to pretty-printed JSON string."""
    return json.dumps(obj, indent=indent, ensure_ascii=False)


def flatten_config_to_text(config: dict[str, Any]) -> str:
    """Flatten nested config dict into searchable text (key:value pairs)."""
    def _flatten(obj: Any, prefix: str = '') -> list[str]:
        items = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    items.extend(_flatten(value, new_prefix))
                else:
                    items.append(f"{new_prefix}:{value}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_prefix = f"{prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    items.extend(_flatten(item, new_prefix))
                else:
                    items.append(f"{new_prefix}:{item}")
        else:
            items.append(f"{prefix}:{obj}")
        
        return items
    
    flattened_items = _flatten(config)
    return ' '.join(flattened_items)


def log(msg: str) -> None:
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


if __name__ == "__main__":
    print("Utils Module Demo")
    print("=" * 80)
    print()
    
    log("Starting utils demo")
    print()
    
    examples_path = os.path.join("data", "examples.json")
    
    try:
        log(f"Loading JSON from {examples_path}")
        examples = load_json(examples_path)
        log(f"✓ Successfully loaded {len(examples)} examples")
        print()
        
        if examples:
            print("First example:")
            print(pretty_json(examples[0]))
            print()
        
    except FileNotFoundError:
        log(f"✗ File not found: {examples_path}")
        log("Creating sample data for demo")
        examples = [
            {
                "id": "ex-001",
                "title": "Sample workflow",
                "description": "A sample workflow configuration",
                "config": {
                    "id": "wf-001",
                    "type": "notification",
                    "condition": {"field": "status", "operator": "==", "value": "complete"},
                    "action": {"type": "email", "recipients": ["user@example.com"]}
                }
            }
        ]
        print()
    
    print("Test 2: Pretty print JSON")
    print("-" * 80)
    sample_obj = {
        "name": "workflow-assistant-rag",
        "version": "1.0.0",
        "features": ["retrieval", "generation", "validation"]
    }
    print(pretty_json(sample_obj, indent=2))
    print()
    
    print("Test 3: Flatten config to searchable text")
    print("-" * 80)
    if examples:
        config = examples[0].get('config', {})
        print(f"Original config:\n{pretty_json(config)}\n")
        
        flattened = flatten_config_to_text(config)
        print(f"Flattened text:\n{flattened}\n")
    
    print("Test 4: Save JSON to file")
    print("-" * 80)
    test_output_path = os.path.join("data", "test_output.json")
    test_data = {
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "message": "This is a test output file"
    }
    
    try:
        log(f"Saving test data to {test_output_path}")
        save_json(test_output_path, test_data)
        log(f"✓ Successfully saved JSON")
        
        log(f"Reading back from {test_output_path}")
        loaded = load_json(test_output_path)
        log(f"✓ Successfully loaded: {loaded.get('message')}")
        
        if os.path.exists(test_output_path):
            os.remove(test_output_path)
            log(f"✓ Cleaned up test file")
    except Exception as e:
        log(f"✗ Error: {e}")
    
    print()
    print("=" * 80)
    log("Utils demo completed")
    
    print("\nAvailable utility functions:")
    print("  - load_json(path): Load JSON from file")
    print("  - save_json(path, obj): Save object to JSON file")
    print("  - pretty_json(obj, indent): Format object as pretty JSON string")
    print("  - flatten_config_to_text(config): Flatten nested config to searchable text")
    print("  - log(msg): Print timestamped log message")
