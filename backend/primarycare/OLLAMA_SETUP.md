# Dual LLM Provider Setup

Your pipeline now supports switching between **LM Studio** and **Ollama** without any code changes!

## How to Use

### 1. Use LM Studio (Default - Current Setup)
Only use environment:
```powershell
# Load the LM Studio configuration (default)
$env:MEDIRAG_LLM_PROVIDER="lmstudio"

# Then start your FastAPI server
python main.py
```

**Expected output:**
```
DEBUG: LLM Provider = lmstudio
DEBUG: LLM Base URL = http://127.0.0.1:1234
DEBUG: LLM Model = qwen2.5-3b-instruct:2
```

---

### 2. Test with Ollama (Separate Testing Path)

#### Step 1: Install Ollama
- Download from: https://ollama.ai
- Install and run it

#### Step 2: Pull the Qwen model
```powershell
ollama pull qwen2.5:3b
```

#### Step 3: Start Ollama server
```powershell
ollama serve
```
(runs on `http://127.0.0.1:11434` by default)

#### Step 4: In a new terminal, switch your pipeline to Ollama
```powershell
$env:MEDIRAG_LLM_PROVIDER="ollama"

# Then start your FastAPI server
python main.py
```

**Expected output:**
```
DEBUG: LLM Provider = ollama
DEBUG: LLM Base URL = http://127.0.0.1:11434
DEBUG: LLM Model = qwen2.5:3b
```

---

### 3. Using `.env` Files (Alternative)
Create a `.env` file in your project root:
```
MEDIRAG_LLM_PROVIDER=ollama
MEDIRAG_OLLAMA_BASE_URL=http://127.0.0.1:11434
MEDIRAG_OLLAMA_MODEL=qwen2.5:3b
S2_API_KEY=your_key_here
```

Then FastAPI will auto-load it (if you use python-dotenv).

---

## Key Points

✅ **LM Studio (Default)** - No changes needed, keeps working as before  
✅ **Ollama (Testing)** - Just set env var and run; same API endpoint compatibility  
✅ **No code changes** - Both use OpenAI-compatible `/v1/chat/completions`  
✅ **Easy switching** - Just change `MEDIRAG_LLM_PROVIDER` env variable  

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Connection refused on 11434` | Make sure Ollama is running (`ollama serve`) |
| `Model not found` | Run `ollama pull qwen2.5:3b` |
| `Still using old provider` | Restart FastAPI after changing env var |
| LM Studio still works | Yes! Both endpoints are configured; switching uses the selected one |

---

## Testing Both Simultaneously

You can run **both** servers at the same time:
1. LM Studio on `http://127.0.0.1:1234`
2. Ollama on `http://127.0.0.1:11434`

Then switch the pipeline by changing `MEDIRAG_LLM_PROVIDER` environment variable between tests.
