# app.py
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain.agents import AgentType, create_pandas_dataframe_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import StructuredTool

# -------------------------------------------------
# 1️⃣ Load env & data
# -------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❗️ Set OPENAI_API_KEY in .env or via Streamlit sidebar.")
    st.stop()

@st.cache_data(ttl="2h")
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # basic cleaning – adapt if your column names differ
    df["over"] = pd.to_numeric(df["over"], errors="coerce")
    df["runs_total"] = pd.to_numeric(df["runs_total"], errors="coerce")
    df["isFour"] = df["isFour"].astype(bool)
    df["isSix"] = df["isSix"].astype(bool)
    df["bowling_style"] = df["bowling_style"].fillna("").astype(str)
    return df

df = load_data("cricket_data.csv")

# -------------------------------------------------
# 2️⃣ LLM & custom tool
# -------------------------------------------------
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4-0125-preview",   # change to gpt-3.5‑turbo if needed
    openai_api_key=OPENAI_API_KEY,
    streaming=True,
)

# custom tool -------------------------------------------------
from utils import runs_by_overs_and_style

runs_tool = StructuredTool.from_function(
    func=lambda overs_start, overs_end, bowling_style: runs_by_overs_and_style(
        df, int(overs_start), int(overs_end), bowling_style
    ),
    name="runs_by_overs_and_style",
    description=(
        "Get total runs, balls, fours and sixes for each batter in a specific "
        "overs range (inclusive) against a given bowling style (e.g. 'pace' or "
        "'spin'). Parameters: overs_start (int), overs_end (int), bowling_style (str)."
    ),
)

# pandas agent -------------------------------------------------
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    extra_tools=[runs_tool],
    allow_dangerous_code=False,
)

# -------------------------------------------------
# 3️⃣ Streamlit UI (chat)
# -------------------------------------------------
st.set_page_config(page_title="🏏 Cricket‑Dataset Chatbot", page_icon="🏏")
st.title("🏏 Cricket‑Dataset Chatbot")

with st.sidebar.expander("📊 Sample data (first 5 rows)"):
    st.dataframe(df.head())

# initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your cricket‑data assistant. Ask me anything about the match data."}
    ]

# render chat
for m in st.session_state.messages:
    if m["role"] == "assistant":
        st.chat_message("assistant").write(m["content"])
    else:
        st.chat_message("user").write(m["content"])

# input box
if prompt := st.chat_input("Your question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # ---- run agent -------------------------------------------------
    with st.chat_message("assistant"):
        handler = StreamlitCallbackHandler(st.container())
        try:
            answer = agent.run(prompt, callbacks=[handler])
        except Exception as e:
            answer = f"❗️ **Error:** {e}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.write(answer)
