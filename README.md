# Workflow Assistant RAG

**Transform natural language into ready-to-use workflow automation configurations.**

Instead of manually writing complex JSON configurations for your workflow automation tools, just describe what you want in plain English. The system retrieves similar examples from its knowledge base, generates valid JSON configs, and validates them against your schema.

## What Problem Does This Solve?

**The Challenge:**
- Workflow automation tools (Zapier, n8n, Apache Airflow, custom systems) require precise JSON configurations
- Writing these configs manually is time-consuming and error-prone
- Remembering field names, structure, and syntax is tedious
- Copying and modifying existing configs leads to inconsistencies

**The Solution:**
- **Type what you want** in plain English: "Send email when task exceeds 2 hours"
- **Get valid JSON** automatically generated and validated
- **Use it anywhere** - workflow tools, APIs, configuration files

## Real-World Use Cases

### 1. Task Management & Ticketing Systems
**You want:** "Escalate high-priority tickets to manager if unresolved for 24 hours"

**You get:** JSON config that your ticketing system can execute
```json
{
  "nodeType": "escalation",
  "conditions": [
    {"field": "priority", "op": "==", "value": "high"},
    {"field": "unresolved_duration", "op": ">", "value": 86400}
  ],
  "actions": [
    {"type": "assign", "params": {"assignee": "manager"}}
  ]
}
```

### 2. DevOps & Monitoring
**You want:** "Send SMS alert when server CPU usage exceeds 80%"

**You get:** Config for your monitoring system (Prometheus, Datadog, etc.)

### 3. Business Process Automation
**You want:** "Auto-approve expense reports under $100, otherwise route to finance team"

**You get:** Config for your approval workflow engine

### 4. Data Pipelines
**You want:** "Retry failed API calls 3 times with exponential backoff"

**You get:** Config for Apache Airflow or custom ETL pipelines

### 5. Customer Support
**You want:** "Create Slack notification when customer satisfaction score drops below 3"

**You get:** Config for your customer feedback system

## How It Works

This is a **Retrieval-Augmented Generation (RAG)** system - it doesn't just guess, it learns from examples.

### The 4-Step Process:

```
1. 🔍 RETRIEVE
   Your query: "notify when task is late"
   → Finds 3 most similar examples from knowledge base
   
2. 📝 BUILD PROMPT  
   → Combines: Schema rules + Retrieved examples + Your request
   → Creates comprehensive instruction for LLM
   
3. 🤖 GENERATE
   → Sends to OpenAI (or shows offline placeholder)
   → Gets back JSON workflow configuration
   
4. ✅ VALIDATE
   → Checks against JSON Schema
   → Ensures all required fields present
   → Reports validation errors or success
```

**Why RAG?** By showing the AI relevant examples first, you get more consistent, accurate outputs that follow your existing patterns.

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Step 1: Clone or Download

```bash
git clone https://github.com/Evalutik/workflow-assistant-rag.git
cd workflow-assistant-rag
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `flask` - Web application framework
- `scikit-learn` - TF-IDF retrieval engine
- `jsonschema` - Config validation
- `requests` - LLM API calls
- `python-dotenv` - Environment variable management
- `pandas`, `numpy` - Data processing

### Step 4: (Optional) Configure OpenAI API

**The app works perfectly without an API key** - it runs in demo mode with simulated responses.

**To enable real AI generation**, create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-api-key-here
```

That's it! The system automatically detects the API key and switches to real AI mode. No code editing needed!

**Get an API key:** [OpenAI Platform](https://platform.openai.com/)

**Note:** In offline mode, the system shows you what prompt would be sent to the AI and generates valid example configs for demonstration.

### Step 5: Verify Installation

```bash
python demo.py
```

You should see the complete pipeline in action with sample outputs.

## Project Structure

```
workflow-assistant-rag/
├── data/
│   ├── examples.json       # 6 example workflow configurations
│   └── schema.json         # JSON Schema for workflow validation
├── retriever.py            # TF-IDF-based example retrieval
├── prompt_builder.py       # LLM prompt construction
├── llm_client.py           # LLM API client (offline mode supported)
├── validate.py             # JSON Schema validation
├── utils.py                # Utility functions (JSON I/O, logging)
├── demo.py                 # Command-line demo script
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Usage

### Option 1: Web Interface (Recommended for Beginners)

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser:**
   ```
   http://127.0.0.1:5000/
   ```

3. **Enter your request:**
   - Type: "Send email notification when task duration exceeds 2 hours"
   - Click "Generate Workflow"

4. **Review the results:**
   - See which examples were retrieved
   - View the generated prompt
   - Check the JSON output
   - Review validation status

5. **Download or copy the JSON** to use in your workflow system

### Option 2: Command-Line Demo

Run the complete pipeline with a sample query:

```bash
python demo.py
```

**What you'll see:**
- Loading of schema and examples
- TF-IDF index building
- Top-3 retrieved examples
- Complete LLM prompt (500+ chars preview)
- Generated JSON configuration
- Validation results with coverage metrics

### Example Queries to Try

Once you have the web app or demo running, try these:

| Natural Language Query | Expected Output Type |
|------------------------|---------------------|
| "Send email when task takes longer than 2 hours" | Notification workflow |
| "Escalate to manager if customer rating is below 3" | Escalation workflow |
| "Retry failed API calls 3 times" | Retry logic workflow |
| "Send SMS alert for critical system errors" | Alert workflow |
| "Auto-assign complex tasks to senior team" | Routing workflow |
| "Generate weekly performance report every Friday" | Scheduled task workflow |

## Understanding the Output

### What You Get Back

**JSON Workflow Configuration** with these components:

```json
{
  "nodeType": "notification",           // Type of automation
  "conditions": [                        // When to trigger
    {
      "field": "duration",
      "op": ">",
      "value": 7200
    }
  ],
  "actions": [                          // What to do
    {
      "type": "email",
      "params": {
        "recipients": ["team@example.com"],
        "subject": "Task Duration Alert"
      }
    }
  ],
  "priority": "high"                    // Optional metadata
}
```

### Where to Use This JSON

1. **Workflow Automation Tools:**
   - Import into n8n, Zapier, Make.com
   - Configure Apache Airflow DAGs
   - Set up automation rules in ticketing systems

2. **Custom Applications:**
   - Save as configuration file (`.json`)
   - Store in database for runtime processing
   - Send to your workflow engine API

3. **CI/CD Pipelines:**
   - Define deployment workflows
   - Configure build triggers
   - Set up automated testing rules

## Customizing for Your Use Case

### Adding Your Own Examples

Edit `data/examples.json` to include your organization's workflow patterns:

```json
[
  {
    "id": "custom-001",
    "title": "Your workflow title",
    "description": "One-sentence description",
    "config": {
      "nodeType": "your_type",
      "conditions": [...],
      "actions": [...]
    }
  }
]
```

**Tip:** The more examples you add, the better the system understands your specific use cases!

### Customizing the Schema

Edit `data/schema.json` to match your workflow system's requirements:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    // Define your fields here
  },
  "required": ["field1", "field2"]  // Mandatory fields
}
```

The validator will ensure all generated configs match your schema.

## Technical Architecture

### Project Structure

```
workflow-assistant-rag/
├── data/
│   ├── examples.json       # Knowledge base (6 workflow examples)
│   └── schema.json         # JSON Schema for validation
├── retriever.py            # TF-IDF similarity search
├── prompt_builder.py       # Prompt construction logic
├── llm_client.py           # OpenAI API client (offline-capable)
├── validate.py             # JSON Schema validation
├── utils.py                # Helper functions
├── demo.py                 # CLI demonstration
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
└── README.md              # Documentation
```

### How Components Work Together

```
┌─────────────────────────────────────────────────────────────┐
│ USER INPUT                                                  │
│ "Send email when task duration > 2 hours"                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ RETRIEVER (retriever.py)                                    │
│ • Loads examples from data/examples.json                    │
│ • Builds TF-IDF index                                       │
│ • Finds 3 most similar examples via cosine similarity       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ PROMPT BUILDER (prompt_builder.py)                          │
│ • Combines: Schema + Retrieved examples + User query        │
│ • Formats structured prompt for LLM                         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ LLM CLIENT (llm_client.py)                                  │
│ • Calls OpenAI API (if key available)                       │
│ • Returns offline placeholder (if no key)                   │
│ • Parses JSON from response                                 │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ VALIDATOR (validate.py)                                     │
│ • Validates JSON against schema (data/schema.json)          │
│ • Computes coverage metric (% required fields)              │
│ • Returns errors or success status                          │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT                                                      │
│ Valid JSON workflow configuration ready to use              │
└─────────────────────────────────────────────────────────────┘
```

### Key Technologies

- **Flask** - Web framework for interactive interface
- **scikit-learn** - TF-IDF vectorization and cosine similarity
- **jsonschema** - JSON Schema validation (Draft-07)
- **OpenAI API** - LLM generation (optional, works offline)
- **python-dotenv** - Environment configuration

### Why TF-IDF?

This prototype uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for simplicity and zero external dependencies:

✅ **Pros:**
- No API calls needed for retrieval
- Fast and deterministic
- Works offline
- Easy to understand

⚠️ **Limitations:**
- Keyword-based (doesn't understand deep semantics)
- Misses synonyms ("delayed" vs "late")

**For Production:** Consider upgrading to:
- Vector embeddings (OpenAI embeddings, Sentence Transformers)
- Vector databases (Pinecone, Weaviate, ChromaDB)
- Semantic search capabilities

## Offline Mode Explained

**What is Offline Mode?**

When you don't have an OpenAI API key set, the system runs in offline mode to let you explore the full pipeline without costs.

**What Happens:**

1. ✅ **Retrieval still works** - Finds similar examples from your knowledge base
2. ✅ **Prompt is built** - Shows you exactly what would be sent to the LLM
3. ⚠️ **No actual API call** - Instead of calling OpenAI, returns a placeholder
4. ✅ **Simulated output** - Creates a fake but valid JSON response for demo
5. ✅ **Validation works** - Checks the simulated output against schema

**Why is this useful?**

- **Learn how RAG works** without spending money
- **Test your examples and schema** before going live
- **See the complete prompt** to understand what the LLM receives
- **Experiment with queries** to refine your knowledge base

**To Enable Real AI:**
1. Get API key from [OpenAI](https://platform.openai.com/)
2. Create `.env` file: `OPENAI_API_KEY=sk-your-key-here`
3. Restart the app - it automatically switches to AI mode!

## Advanced Usage

### Running Individual Components

**Test retrieval only:**
```bash
python retriever.py
```

**Test prompt building:**
```bash
python prompt_builder.py
```

**Test validation:**
```bash
python validate.py
```

**Test LLM client:**
```bash
python llm_client.py
```

Each module has a `__main__` block demonstrating its functionality.

### Integrating with Your System

**Example: Using in a Python script**

```python
from retriever import load_examples, build_index, get_top_k
from prompt_builder import build_prompt
from llm_client import call_llm, parse_llm_text_to_json
from validate import validate_output
from utils import load_json

# Load your data
schema = load_json("data/schema.json")
examples = load_examples("data/examples.json")

# Build index once
vectorizer, matrix, _ = build_index(examples)

# Process user query
query = "Send alert when CPU > 80%"
retrieved = get_top_k(query, vectorizer, matrix, examples, k=3)
prompt = build_prompt(query, retrieved, schema)

# Get LLM response
result = call_llm(prompt)
config, error = parse_llm_text_to_json(result.get("text", "{}"))

# Validate
is_valid, errors = validate_output(config, schema)

if is_valid:
    # Use the config in your system
    print("Generated config:", config)
else:
    print("Validation errors:", errors)
```

## Troubleshooting

### Common Issues

**Issue:** `FileNotFoundError: data/examples.json`
- **Solution:** Make sure you're running commands from the project root directory
- **Check:** `ls data/` should show `examples.json` and `schema.json`

**Issue:** `ModuleNotFoundError: No module named 'flask'`
- **Solution:** Activate virtual environment and install dependencies
- **Run:** `pip install -r requirements.txt`

**Issue:** Import errors or missing modules
- **Solution:** Ensure virtual environment is activated
- **Windows:** `venv\Scripts\activate`
- **Linux/Mac:** `source venv/bin/activate`

**Issue:** Schema validation always fails
- **Solution:** Check that your `data/schema.json` matches the structure in examples
- **Verify:** Required fields in schema match your example configs

**Issue:** Retrieval returns irrelevant examples
- **Solution:** Add more diverse examples to `data/examples.json`
- **Tip:** Include examples with varied terminology and use cases

**Issue:** Web app shows "offline mode" but I have an API key
- **Solution:** Uncomment the API request code in `llm_client.py` (see line ~45)
- **Note:** By default, API calls are commented out for safety

### Getting Help

- Check the inline comments in each Python file
- Run individual modules to test components
- Review `demo.py` for a working example of the full pipeline

## Production Deployment Considerations

This is a **prototype/proof-of-concept**. For production use:

### Recommended Upgrades

**1. Retrieval System:**
- Replace TF-IDF with vector embeddings (OpenAI, Cohere, Sentence-BERT)
- Use vector database (Pinecone, Weaviate, ChromaDB)
- Implement hybrid search (keyword + semantic)
- Add caching for frequent queries

**2. LLM Integration:**
- Add support for multiple providers (Anthropic, Azure OpenAI, local models)
- Implement retry logic with exponential backoff
- Add rate limiting and request queuing
- Enable streaming responses for better UX
- Monitor token usage and costs

**3. Validation:**
- Add business logic validators beyond JSON Schema
- Implement cross-field validation rules
- Create custom error messages for your domain
- Add workflow simulation/dry-run capability

**4. Production Features:**
- User authentication and authorization
- Usage analytics and logging
- A/B testing for prompt variations
- Feedback loop to improve examples
- Version control for schemas and examples
- API endpoints for programmatic access

**5. Infrastructure:**
- Containerization (Docker)
- Horizontal scaling
- Load balancing
- Monitoring and alerting
- Database for storing generated configs

## Contributing

This is a demonstration project showing RAG concepts. Feel free to:
- Fork and adapt for your specific use case
- Add more sophisticated retrieval methods
- Integrate with other LLM providers
- Extend validation logic
- Improve the web interface

## License

MIT License - See LICENSE file for details

---

## Quick Reference Card

### Setup (One-Time)
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run Web App
```bash
python app.py
# Open http://127.0.0.1:5000/
```

### Run Demo
```bash
python demo.py
```

### Add API Key (Optional)
Create `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

### Customize Examples
Edit `data/examples.json` with your workflows

### Customize Schema
Edit `data/schema.json` to match your system

---

**Questions?** Review the inline comments in the Python files or run `python <module>.py` to see demonstrations of individual components.
