# PDF Research Chat

A local RAG chatbot for the PDF files in `PDF Files/`.

## Run it

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL printed by Streamlit.

Run the automated tests from the project root with:

```powershell
python -m pytest
```

## Project layout

```text
QA System/
├── app.py
├── test_adapter.py
├── PDF Files/
├── tests/
│   └── test_rag_system.py
├── performance_test.py
├── stress_test.py
├── test_cases.json
├── .env
└── requirements.txt
```

The app extracts PDF text, creates overlapping page chunks, retrieves relevant chunks with TF-IDF, and shows source filenames and page numbers. It works immediately in **Extractive** mode. For generated answers, select **OpenAI** and set `OPENAI_API_KEY`, or select **Ollama** after starting a local Ollama model.

Environment variables:

- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_MODEL`: optional OpenAI model, default `gpt-4o-mini`
- `OLLAMA_MODEL`: optional Ollama model, default `llama3.2`
- `OLLAMA_BASE_URL`: optional Ollama-compatible URL, default `http://localhost:11434/v1`