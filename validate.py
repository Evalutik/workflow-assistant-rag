"""JSON Schema validation and coverage metrics."""

import json
from typing import Any
from jsonschema import validate, ValidationError, SchemaError


def validate_output(candidate: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate JSON against schema. Returns (is_valid, error_list)."""
    errors = []
    
    try:
        validate(instance=candidate, schema=schema)
        return True, []
        
    except ValidationError as e:
        error_msg = f"Validation error at path '{'.'.join(str(p) for p in e.path)}': {e.message}"
        errors.append(error_msg)
        
        if e.context:
            for suberror in e.context:
                sub_msg = f"  - Sub-error: {suberror.message}"
                errors.append(sub_msg)
        
        return False, errors
        
    except SchemaError as e:
        errors.append(f"Schema error: {e.message}")
        return False, errors
        
    except Exception as e:
        errors.append(f"Unexpected validation error: {str(e)}")
        return False, errors


def coverage_metric(candidate: dict[str, Any], schema: dict[str, Any]) -> float:
    """Calculate fraction of required fields present in candidate (0.0 to 1.0)."""
    required_fields = schema.get('required', [])
    
    if not required_fields:
        return 1.0
    
    present_count = sum(1 for field in required_fields if field in candidate)
    return present_count / len(required_fields)


if __name__ == "__main__":
    import os
    
    print("Validation Module Demo")
    print("=" * 80)
    print()
    
    schema_path = os.path.join("data", "schema.json")
    
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        print(f"✓ Loaded schema from {schema_path}")
        print(f"Schema:\n{json.dumps(schema, indent=2)}\n")
    except FileNotFoundError:
        print(f"✗ Schema file not found: {schema_path}")
        print("Using a sample schema for demo purposes\n")
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
        print(f"Sample schema:\n{json.dumps(schema, indent=2)}\n")
    
    print("=" * 80)
    print()
    
    print("Test 1: Valid candidate")
    print("-" * 80)
    valid_candidate = {
        "id": "wf-001",
        "type": "notification",
        "condition": {"field": "duration", "operator": ">", "value": 120},
        "action": {"type": "email", "recipients": ["admin@example.com"]},
        "priority": "high"
    }
    
    print(f"Candidate:\n{json.dumps(valid_candidate, indent=2)}\n")
    
    is_valid, errors = validate_output(valid_candidate, schema)
    coverage = coverage_metric(valid_candidate, schema)
    
    if is_valid:
        print("✓ Validation: PASSED")
    else:
        print("✗ Validation: FAILED")
        for error in errors:
            print(f"  {error}")
    
    print(f"Coverage: {coverage:.2%} of required fields present\n")
    
    print("Test 2: Invalid candidate (missing required field)")
    print("-" * 80)
    invalid_candidate_1 = {
        "id": "wf-002",
        "type": "notification",
        "priority": "medium"
    }
    
    print(f"Candidate:\n{json.dumps(invalid_candidate_1, indent=2)}\n")
    
    is_valid, errors = validate_output(invalid_candidate_1, schema)
    coverage = coverage_metric(invalid_candidate_1, schema)
    
    if is_valid:
        print("✓ Validation: PASSED")
    else:
        print("✗ Validation: FAILED")
        for error in errors:
            print(f"  {error}")
    
    print(f"Coverage: {coverage:.2%} of required fields present\n")
    
    print("Test 3: Invalid candidate (wrong enum value)")
    print("-" * 80)
    invalid_candidate_2 = {
        "id": "wf-003",
        "type": "notification",
        "action": {"type": "slack"},
        "priority": "critical"
    }
    
    print(f"Candidate:\n{json.dumps(invalid_candidate_2, indent=2)}\n")
    
    is_valid, errors = validate_output(invalid_candidate_2, schema)
    coverage = coverage_metric(invalid_candidate_2, schema)
    
    if is_valid:
        print("✓ Validation: PASSED")
    else:
        print("✗ Validation: FAILED")
        for error in errors:
            print(f"  {error}")
    
    print(f"Coverage: {coverage:.2%} of required fields present\n")
    
    print("Test 4: Partial candidate (2 of 3 required fields)")
    print("-" * 80)
    partial_candidate = {
        "id": "wf-004",
        "type": "alert"
    }
    
    print(f"Candidate:\n{json.dumps(partial_candidate, indent=2)}\n")
    
    is_valid, errors = validate_output(partial_candidate, schema)
    coverage = coverage_metric(partial_candidate, schema)
    
    if is_valid:
        print("✓ Validation: PASSED")
    else:
        print("✗ Validation: FAILED")
        for error in errors:
            print(f"  {error}")
    
    print(f"Coverage: {coverage:.2%} of required fields present\n")
    
    print("=" * 80)
    print("\nSummary:")
    print("- validate_output() checks full schema compliance")
    print("- coverage_metric() measures required field coverage (0.0 to 1.0)")
    print("- Both functions handle errors gracefully without raising exceptions")
