# 🏏 Cricket‑Dataset Chatbot

A conversational chatbot that answers any question about a cricket match‑by‑match CSV
(e.g. “most runs”, “top wicket‑takers”, or “runs in overs 16‑20 vs pace”).

- **Backend** – pandas + LangChain `create_pandas_dataframe_agent`
- **LLM** – OpenAI (GPT‑4 or GPT‑3.5) via `langchain-openai`
- **UI** – Streamlit chat interface
- **Custom tool** – `runs_by_overs_and_style` for overs‑range & bowling‑style queries.

> **Live demo** – after you deploy (see instructions below) you’ll get a URL such as  
> `https://cricket-chatbot-<your‑github‑username>.streamlit.app`.

## Quick start (local)

```bash
git clone https://github.com/<your‑username>/cricket-chatbot.git
cd cricket-chatbot
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# put your CSV at the root as cricket_data.csv
echo "OPENAI_API_KEY=sk-xxxx" > .env          # or add the key in the Streamlit sidebar
streamlit run app.py
