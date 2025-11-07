"""
LLM client with automatic API detection.

SETUP:
1. (Optional) Create .env file with: OPENAI_API_KEY=sk-your-key-here
2. If API key exists → automatically calls OpenAI API
3. If no API key → runs in offline demo mode with simulated responses

No code editing needed - just add the API key when ready!
"""

import os
import json
import re
from typing import Any, Optional
from dotenv import load_dotenv
import requests

load_dotenv()


def call_llm(prompt: str, model: str = "gpt-4") -> dict[str, Any]:
    """Call LLM API automatically if API key is set, otherwise return offline placeholder."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return {
            "offline": True,
            "prompt": prompt,
            "note": "OPENAI_API_KEY not set. Add it to .env file to enable real AI responses.",
            "text": ""
        }
    
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        response_text = data["choices"][0]["message"]["content"]
        return {"text": response_text, "raw": data, "offline": False}
        
    except requests.exceptions.RequestException as e:
        return {"error": True, "message": f"API request failed: {str(e)}", "prompt": prompt, "text": ""}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {"error": True, "message": f"Failed to parse API response: {str(e)}", "prompt": prompt, "text": ""}


def parse_llm_text_to_json(text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Parse LLM output to JSON, handling code fences and whitespace. Returns (dict, error)."""
    if not text:
        return None, "Empty text provided"
    
    cleaned = text.strip()
    
    # Remove markdown code fences
    fence_pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
    match = re.match(fence_pattern, cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    
    try:
        parsed = json.loads(cleaned)
        
        if not isinstance(parsed, dict):
            return None, f"Parsed JSON is not an object (got {type(parsed).__name__})"
        
        return parsed, None
        
    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing error: {str(e)}"
        if e.pos:
            start = max(0, e.pos - 50)
            end = min(len(cleaned), e.pos + 50)
            context = cleaned[start:end]
            error_msg += f"\nContext near error: ...{context}..."
        return None, error_msg


if __name__ == "__main__":
    print("LLM Client Demo")
    print("=" * 80)
    print()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✓ OPENAI_API_KEY is set (length: {len(api_key)} chars)")
    else:
        print("✗ OPENAI_API_KEY not set - running in OFFLINE mode")
    print()
    
    print("Test 1: Calling LLM with sample prompt")
    print("-" * 80)
    sample_prompt = "Generate a JSON object with fields: id, name, status"
    
    result = call_llm(sample_prompt, model="gpt-4")
    
    if result.get("offline"):
        print("Mode: OFFLINE")
        print(f"Note: {result.get('note')}")
        print(f"\nPrompt (first 200 chars):\n{result['prompt'][:200]}...")
    elif result.get("error"):
        print("Mode: ERROR")
        print(f"Error: {result.get('message')}")
    else:
        print("Mode: ONLINE")
        print(f"Response: {result.get('text')}")
    
    print("\n")
    
    print("Test 2: Parsing valid JSON response")
    print("-" * 80)
    valid_json = '{"id": "wf-001", "type": "notification", "status": "active"}'
    parsed, error = parse_llm_text_to_json(valid_json)
    
    if parsed:
        print("✓ Successfully parsed:")
        print(json.dumps(parsed, indent=2))
    else:
        print(f"✗ Parse failed: {error}")
    
    print("\n")
    
    print("Test 3: Parsing JSON wrapped in code fences")
    print("-" * 80)
    fenced_json = '''```json
{
  "id": "wf-002",
  "type": "alert",
  "priority": "high"
}
```'''
    parsed, error = parse_llm_text_to_json(fenced_json)
    
    if parsed:
        print("✓ Successfully parsed (fences removed):")
        print(json.dumps(parsed, indent=2))
    else:
        print(f"✗ Parse failed: {error}")
    
    print("\n")
    
    print("Test 4: Parsing invalid JSON")
    print("-" * 80)
    invalid_json = '{"id": "wf-003", "type": "alert", missing_quote}'
    parsed, error = parse_llm_text_to_json(invalid_json)
    
    if parsed:
        print("✓ Successfully parsed:")
        print(json.dumps(parsed, indent=2))
    else:
        print(f"✗ Parse failed (expected): {error}")
    
    print("\n")
    print("=" * 80)
    print("\nTo enable real API calls:")
    print("1. Add OPENAI_API_KEY to .env file")
    print("2. Uncomment the API request code in call_llm()")
    print("3. Comment out the placeholder return statement")
