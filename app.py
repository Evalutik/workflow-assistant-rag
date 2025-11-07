"""
Flask web app for workflow-assistant-rag.

QUICK START:
1. pip install -r requirements.txt
2. (Optional) Create .env with: OPENAI_API_KEY=your-key
3. python app.py
4. Open http://127.0.0.1:5000/

Without API key, runs in offline mode showing prompts instead of calling LLM.
For CLI demo: python demo.py
"""

import os
import json
from flask import Flask, request, render_template_string, jsonify, session
from retriever import load_examples, build_index, get_top_k
from prompt_builder import build_prompt
from llm_client import call_llm, parse_llm_text_to_json
from validate import validate_output, coverage_metric
from utils import load_json

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global cache (loaded once at startup)
SCHEMA = None
EXAMPLES = None
VECTORIZER = None
TFIDF_MATRIX = None


def initialize_data():
    """Load schema and examples, build index on startup."""
    global SCHEMA, EXAMPLES, VECTORIZER, TFIDF_MATRIX
    
    schema_path = os.path.join("data", "schema.json")
    examples_path = os.path.join("data", "examples.json")
    
    try:
        SCHEMA = load_json(schema_path)
        print(f"✓ Loaded schema from {schema_path}")
    except FileNotFoundError:
        print(f"⚠ Schema not found, using default schema")
        SCHEMA = {
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
        EXAMPLES = load_examples(examples_path)
        print(f"✓ Loaded {len(EXAMPLES)} examples from {examples_path}")
    except FileNotFoundError:
        print(f"⚠ Examples not found, using default examples")
        EXAMPLES = [
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
            }
        ]
    
    print("Building retrieval index...")
    VECTORIZER, TFIDF_MATRIX, _ = build_index(EXAMPLES)
    print(f"✓ Index ready with {len(EXAMPLES)} examples\n")


# HTML Templates
HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Workflow Assistant RAG</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            max-width: 900px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
            box-sizing: border-box;
        }
        button {
            background: #0066cc;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background: #0052a3;
        }
        .info-box {
            background: #e7f3ff;
            border-left: 4px solid #0066cc;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .examples {
            margin-top: 20px;
        }
        .example-item {
            background: #f9f9f9;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            font-size: 14px;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }
        a {
            color: #0066cc;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Workflow Assistant RAG</h1>
        <p class="subtitle">Retrieval-Augmented Generation for Workflow Configurations</p>
        
        <div class="info-box">
            <strong>ℹ️ How it works:</strong>
            <ol style="margin: 10px 0; padding-left: 20px;">
                <li>Enter a natural language description of your workflow</li>
                <li>System retrieves similar examples from the knowledge base</li>
                <li>Builds a prompt with schema + examples + your request</li>
                <li>Generates a JSON workflow configuration</li>
                <li>Validates output against the JSON schema</li>
            </ol>
            <p style="margin-top: 10px; color: #666; font-size: 13px;">
                <strong>Note:</strong> Currently running in <strong>{% if api_key_set %}online mode with AI{% else %}offline demo mode{% endif %}</strong>.
                {% if not api_key_set %}
                To enable real AI responses, create a <code>.env</code> file with <code>OPENAI_API_KEY=sk-your-key-here</code>
                {% endif %}
            </p>
        </div>
        
        <form method="POST" action="/run">
            <label for="query" style="display: block; margin-bottom: 8px; font-weight: 500;">
                Enter your workflow request:
            </label>
            <textarea 
                name="query" 
                id="query" 
                rows="6" 
                placeholder="Example: send email notification when a task takes longer than 2 hours"
                required
            ></textarea>
            <button type="submit">🚀 Generate Workflow</button>
        </form>
        
        <div class="examples">
            <p style="font-weight: 500; margin-bottom: 10px;">Example queries to try:</p>
            <div class="example-item">📧 Send email alert when task status changes to failed</div>
            <div class="example-item">⏰ Notify team on Slack if deadline is missed</div>
            <div class="example-item">🔔 Create alert when resource usage exceeds 80%</div>
        </div>
        
        <div class="footer">
            <p>Loaded {{ num_examples }} examples | 
               <a href="https://github.com/Evalutik/workflow-assistant-rag" target="_blank">View on GitHub</a>
            </p>
        </div>
    </div>
</body>
</html>
"""

RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Results - Workflow Assistant RAG</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            max-width: 1200px;
            margin: 30px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        h1, h2 {
            color: #333;
        }
        h2 {
            border-bottom: 2px solid #0066cc;
            padding-bottom: 10px;
            margin-top: 30px;
        }
        .query-box {
            background: #e7f3ff;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
            border-left: 4px solid #0066cc;
        }
        .retrieved-item {
            background: #f9f9f9;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 3px solid #28a745;
        }
        .score {
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }
        details {
            margin: 15px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
        }
        summary {
            cursor: pointer;
            font-weight: 500;
            padding: 5px;
            user-select: none;
        }
        summary:hover {
            background: #f0f0f0;
        }
        pre {
            background: #f4f4f4;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 13px;
            line-height: 1.5;
        }
        .status-valid {
            color: #28a745;
            font-weight: bold;
        }
        .status-invalid {
            color: #dc3545;
            font-weight: bold;
        }
        .error-list {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
        }
        .coverage {
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 10px 0;
        }
        .offline-notice {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }
        .button-group {
            margin: 20px 0;
        }
        .btn {
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 4px;
            text-decoration: none;
            font-weight: 500;
            border: none;
            cursor: pointer;
        }
        .btn-primary {
            background: #0066cc;
            color: white;
        }
        .btn-primary:hover {
            background: #0052a3;
        }
        .btn-success {
            background: #28a745;
            color: white;
        }
        .btn-success:hover {
            background: #218838;
        }
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        .btn-secondary:hover {
            background: #5a6268;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Workflow Generation Results</h1>
        
        <h2>📝 User Query</h2>
        <div class="query-box">
            {{ query }}
        </div>
        
        <h2>🔍 Retrieved Examples (Top {{ retrieved|length }})</h2>
        {% for item in retrieved %}
        <div class="retrieved-item">
            <span class="score">{{ "%.4f"|format(item.score) }}</span>
            <strong>{{ item.id }}</strong> - {{ item.title }}
            <br><span style="color: #666; font-size: 14px;">{{ item.description }}</span>
        </div>
        {% endfor %}
        
        <h2>💬 LLM Prompt</h2>
        <details>
            <summary>▶ Click to show/hide full prompt ({{ prompt_length }} characters)</summary>
            <pre>{{ prompt }}</pre>
        </details>
        
        {% if offline_mode %}
        <div class="offline-notice">
            <strong>⚠️ OFFLINE MODE - SIMULATED RESPONSE</strong>
            <br><br>
            {{ offline_note }}
            <br><br>
            <strong>Note:</strong> The response below is a <u>simulated placeholder</u> generated automatically to demonstrate the system. 
            It is NOT from a real AI - it's just an example showing what a valid workflow config looks like.
            <br><br>
            <em>To get real AI-generated configs: Create a <code>.env</code> file in the project root and add <code>OPENAI_API_KEY=sk-your-key-here</code></em>
        </div>
        {% endif %}
        
        <h2>🤖 {% if offline_mode %}Simulated Response (Demo Placeholder){% else %}LLM Response{% endif %}</h2>
        {% if llm_response %}
        <pre>{{ llm_response }}</pre>
        {% else %}
        <p style="color: #dc3545;">No response generated</p>
        {% endif %}
        
        <h2>✅ Validation Results</h2>
        {% if is_valid %}
        <div class="coverage">
            <p class="status-valid">✅ Status: VALID</p>
            <p>The generated workflow configuration conforms to the schema.</p>
        </div>
        {% else %}
        <div class="error-list">
            <p class="status-invalid">❌ Status: INVALID</p>
            {% if errors %}
            <p><strong>Errors:</strong></p>
            <ul>
                {% for error in errors %}
                <li>{{ error }}</li>
                {% endfor %}
            </ul>
            {% endif %}
        </div>
        {% endif %}
        
        <div class="coverage">
            <p><strong>📊 Coverage Metric:</strong> {{ "%.1f"|format(coverage * 100) }}%</p>
            <p>{{ "%.0f"|format(coverage * 100) }}% of required fields are present in the output.</p>
        </div>
        
        {% if parsed_json %}
        <h2>📄 Generated JSON Output</h2>
        <details open>
            <summary>▶ JSON Configuration</summary>
            <pre>{{ parsed_json }}</pre>
        </details>
        {% endif %}
        
        <div class="button-group">
            <a href="/" class="btn btn-primary">← New Query</a>
            {% if parsed_json %}
            <a href="/download" class="btn btn-success">⬇️ Download JSON</a>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""


@app.route('/')
def home():
    """Render the home page with the query form."""
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    return render_template_string(HOME_TEMPLATE, num_examples=len(EXAMPLES), api_key_set=api_key_set)


@app.route('/run', methods=['POST'])
def run_pipeline():
    """Run the complete RAG pipeline and display results."""
    query = request.form.get('query', '').strip()
    
    if not query:
        return "Error: No query provided", 400
    
    k = 3
    retrieved = get_top_k(query, VECTORIZER, TFIDF_MATRIX, EXAMPLES, k=k)
    
    retrieved_data = [
        {
            'id': example['id'],
            'title': example.get('title', 'No title'),
            'description': example.get('description', 'No description'),
            'score': score
        }
        for example, score in retrieved
    ]
    
    prompt = build_prompt(query, retrieved, SCHEMA)
    llm_result = call_llm(prompt, model="gpt-4")
    
    offline_mode = llm_result.get('offline', False)
    offline_note = llm_result.get('note', '')
    
    if offline_mode:
        # Generate a realistic simulated output that matches the schema
        simulated_output = {
            "nodeType": "notification",
            "conditions": [
                {
                    "field": "status",
                    "op": "==",
                    "value": "failed"
                }
            ],
            "actions": [
                {
                    "type": "email",
                    "params": {
                        "recipients": ["admin@example.com", "team@example.com"],
                        "subject": "Task Status Alert",
                        "body": "A task status has changed to failed"
                    }
                }
            ],
            "priority": "high",
            "enabled": True
        }
        llm_response_text = json.dumps(simulated_output, indent=2)
    else:
        llm_response_text = llm_result.get('text', '{}')
    
    parsed_output, parse_error = parse_llm_text_to_json(llm_response_text)
    
    if parse_error:
        parsed_output = {}
    
    is_valid, errors = validate_output(parsed_output, SCHEMA)
    coverage = coverage_metric(parsed_output, SCHEMA)
    
    session['last_output'] = parsed_output
    
    return render_template_string(
        RESULTS_TEMPLATE,
        query=query,
        retrieved=retrieved_data,
        prompt=prompt,
        prompt_length=len(prompt),
        offline_mode=offline_mode,
        offline_note=offline_note,
        llm_response=llm_response_text,
        is_valid=is_valid,
        errors=errors,
        coverage=coverage,
        parsed_json=json.dumps(parsed_output, indent=2) if parsed_output else None
    )


@app.route('/download')
def download_json():
    """Download the last generated JSON output."""
    output = session.get('last_output', {})
    if not output:
        return jsonify({"error": "No output available"}), 404
    return jsonify(output)


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("  WORKFLOW ASSISTANT RAG - Flask Web App")
    print("=" * 80 + "\n")
    
    initialize_data()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✓ OPENAI_API_KEY detected (length: {len(api_key)} chars)")
        print("  LLM calls will be enabled (if uncommented in llm_client.py)\n")
    else:
        print("⚠ OPENAI_API_KEY not set")
        print("  Running in OFFLINE mode - will show prompts instead of calling API\n")
    
    print("Starting Flask server...")
    print("Access the app at: http://127.0.0.1:5000/")
    print("\nPress CTRL+C to stop the server\n")
    print("=" * 80 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
