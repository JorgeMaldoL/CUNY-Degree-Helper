# Dream 4 Degree

Find the right CUNY program with a fast, local RAG search and a simple Streamlit UI.

## Quick start

```bash
pip install -r requirements.txt
streamlit run frontend/frontend.py
```

## Config
- Set `ANTHROPIC_API_KEY` (env var or Streamlit secrets) for Advisor answers.
- Data is in `data/`; the Chroma index is persisted under `backend/chroma_db`.

## Deploy
- Streamlit Community Cloud → Main file path: `frontend/frontend.py`.


