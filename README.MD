# 🧠Structured JSON Extractor: Unstructured Text → Structured JSON

> Convert complex unstructured documents into valid structured JSON using LLMs like GPT-4.1 and O4 Mini with schema adherence and validation.

---

## 🔗 App Access

**Streamlit App**: [https://jason-extract-gen.streamlit.app/](https://jason-extract-gen.streamlit.app/)

---

## 🚀 How to Use the App

1. Visit the app link above.
2. Upload a valid JSON schema.
3. Upload a text file (`.txt` or `.pdf`) that contains unstructured input.
4. Choose your model from the dropdown:
   - 🧠 **GPT 4.1** — Best for simpler, smaller schemas.
   - 🧩 **GPT O4 Mini** — Handles long text + large complex schemas (up to 100k tokens).
5. Click **Generate JSON**.
6. Wait while the system:
   - Validates your schema.
   - Estimates token size.
   - Prompts the model up to 3 times until output is schema-valid.
7. View the structured JSON.
8. Download the JSON and logs.

---

## 🔌 API Setup - Application Steps

### ▶️ Running the API

```bash
git clone <https://github.com/cosmoavijeet07/JSON-generator.git>
cd JSON-generator
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env

uvicorn fastapi_app:app --reload --port 8000
```

---

## 📡 API Endpoint

### POST `/extract-json/`

**Description**: Generate a structured JSON from unstructured input using a schema and selected model.

#### 🔸 Input: `multipart/form-data`
| Field   | Type        | Required | Description                   |
|---------|-------------|----------|-------------------------------|
| schema  | File (.json)| ✅        | Valid JSON Schema             |
| text    | File (.txt) | ✅        | Input text document           |
| model   | String      | ❌        | Model ID (default GPT 4.1)    |

#### 🔹 Sample `curl` Request
```bash
curl -X POST http://localhost:8000/extract-json/ \
  -F "schema=@schema.json" \
  -F "text=@input.txt" \
  -F "model=gpt-4.1-2025-04-14"
```

#### ✅ Success Response
```json
{
  "session_id": "b95c...",
  "token_estimate": 8372,
  "attempts": 2,
  "json_result": { [...]
  }
}
```

#### ❌ Failure Response
```json
{
  "error": "Validation failed after 3 attempts: Missing required field",
  "session_id": "b95c...",
  "token_estimate": 10423
}
```

---

## 🧱 Architecture Diagram
![alt text](image.png)


---

## 🧰 Technologies Used

- **LLM APIs**: OpenAI GPT-4.1 & GPT-O4 Mini
- **Prototype**: Streamlit
- **API Service**: FastAPI
- **Validation**: JSON Schema + `jsonschema`
- **Token Estimation**: tiktoken
- **Prompting**: Custom few-shot + retry
- **Logging**: Per-session logs stored in `logs/`

---

## 📁 Folder Structure

```text
JSON GENERATOR/
├── core/                         # Core logic modules
│   ├── json_extractor.py
│   ├── llm_interface.py
│   ├── logger_service.py
│   ├── prompt_engine.py
│   ├── schema_validator.py
│   ├── session_manager.py
│   ├── token_estimator.py
│
├── logs/                         # Session logs
│
├── Sample Inputs with output/    # Examples & output
│
├── .env                          # OpenAI keys
├── .gitignore
├── app.py                        # Streamlit frontend
├── fastapi_app.py                # FastAPI backend
├── requirements.txt
└── README.md
```

---

## 📚 CORE files

Each module in the `core/` folder is fully commented and structured.

| File | Purpose |
|------|---------|
| `prompt_engine.py` | Schema-aware prompts with multi-shot + retry handling |
| `json_extractor.py` | Extracts clean JSON from LLM output and validates against schema |
| `llm_interface.py` | Manages calls to OpenAI API with retry-safe abstraction |
| `token_estimator.py` | Uses `tiktoken` to estimate total tokens for cost/safety |
| `logger_service.py` | Writes structured per-session logs to disk |

---
🧾 View detailed documentation [doc](docdotcom).

## 👤 Author & Support

Developed by `Avijeet Palit`
📫 Reach out: [avijeet.palit07@gmail.com]
