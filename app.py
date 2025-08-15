"""
app.py
-------

Streamlit UI + LangChain pandas‑agent chatbot for a cricket match CSV.
All custom aggregations (batter, bowler, team, win‑probability, etc.) are
implemented as pure‑Python functions in **utils.py** and exposed to the LLM
via `StructuredTool`.
"""

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
# -------------------------------------------------
# ✅  Correct imports for LangChain with Groq
# -------------------------------------------------
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# -------------------------------------------------
# 1️⃣  Load environment variables & CSV data
# -------------------------------------------------
load_dotenv()                     # reads .env (or the GitHub secret)

# Load environment variables
load_dotenv()

# Check for Groq API key in environment variables or GitHub secrets
groq_api_key = os.environ.get('GROQ_API_KEY')

if not groq_api_key:
    st.warning("⚠️ GROQ_API_KEY not found in environment variables.")
    st.info("""
    **To use Groq API (FREE & FAST!):**

    For local development:
    1. Get a FREE API key from https://console.groq.com/
    2. Create a `.env` file in the project root with:
       ```
       GROQ_API_KEY=your_api_key_here
       ```
    3. Restart the app
    
    For deployment:
    1. Add GROQ_API_KEY as a repository secret in your GitHub repository
    2. The GitHub Actions workflow will automatically use it
    
    **Benefits of Groq:**
    - 🚀 Ultra-fast inference (fastest in market)
    - 💰 FREE tier: 14,400 requests/day
    - 📊 Paid tier: $0.27-$0.59 per 1M tokens (70% cheaper than OpenAI)
    - 🤖 Access to Llama 3.1 70B, Mixtral 8x7B, Gemma models
    """)
    st.stop()

# Set the API key for Groq
os.environ['GROQ_API_KEY'] = groq_api_key

st.success("✅ Groq API configured successfully! 🚀")


@st.cache_data(ttl="2h")   # cache the DataFrame for 2 hours – speeds up reloads
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV and perform a minimal clean‑up.
    Adjust column names here if your CSV differs.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Cast numeric columns (invalid entries become NaN)
    df["over"] = pd.to_numeric(df["over"], errors="coerce")
    df["runs_total"] = pd.to_numeric(df["runs_total"], errors="coerce")
    df["runs_batter"] = pd.to_numeric(df["runs_batter"], errors="coerce")
    df["balls_faced"] = pd.to_numeric(df["balls_faced"], errors="coerce")

    # Boolean flags (0/1 → False/True)
    df["isFour"] = df["isFour"].astype(bool)
    df["isSix"] = df["isSix"].astype(bool)
    df["isWicket"] = df["isWicket"].astype(bool)

    # Ensure bowling_style exists and is a string
    df["bowling_style"] = df["bowling_style"].fillna("").astype(str)
    df["bowling_type"] = df["bowling_type"].fillna("").astype(str)
    df["bat_hand"] = df["bat_hand"].fillna("").astype(str)

    return df


# ----------------------------------------------------------------------
# Load the CSV that sits in the repo root (rename if needed)
# ----------------------------------------------------------------------
df = load_data("cricket_data.csv")

# -------------------------------------------------
# 2️⃣  LLM & custom tool definition (expanded)
# -------------------------------------------------
# Initialize Groq LLM (Fast & Cheap!)
llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama3-70b-8192",  # Best performance model
    temperature=0,  # Deterministic responses
    max_retries=3,
    timeout=30,
    # Optional: Use different models based on need
    # "llama3-8b-8192" for faster responses
    # "mixtral-8x7b-32768" for complex reasoning  
    # "gemma-7b-it" for lightweight tasks
)

# ------------------------------------------------------------------
# Import all helper functions from utils.py
# ------------------------------------------------------------------
from utils import (
    # Basic Statistics
    get_most_runs,
    get_most_wickets,
    get_most_fours_and_sixes,
    
    # Advanced Batting Analysis
    runs_by_overs_and_style,
    best_batters_death_overs,
    best_batters_vs_batting_hand,
    
    # Advanced Bowling Analysis
    bowler_conceded_runs,
    bowler_wickets,
    best_bowlers_death_overs,
    best_bowlers_vs_batting_hand,
    bowler_economy,
    
    # Fielding Analysis
    get_fielding_stats,
    
    # Partnership Analysis
    get_best_partnerships,
    
    # Team Analysis
    team_total_runs,
    team_wickets_lost,
    team_run_rate,
    
    # Match Analysis
    get_match_results,
    get_venue_stats,
    
    # Advanced Analytics
    get_powerplay_stats,
    get_middle_overs_stats,
    get_death_overs_stats,
    
    # Team-Specific Analysis
    get_team_best_batters,
    get_team_best_bowlers,
    get_team_overall_stats,
    
    # Utility Functions
    top_strike_rate,
    win_probability,
    best_partnerships,
)

# ------------------------------------------------------------------
# Wrap each helper in a StructuredTool so the LLM can call it.
# ------------------------------------------------------------------
tools = []

# ------------------- Basic Statistics Tools -------------------------
most_runs_tool = StructuredTool.from_function(
    func=get_most_runs,
    name="get_most_runs",
    description="Get the top run scorers in the tournament"
)

most_wickets_tool = StructuredTool.from_function(
    func=get_most_wickets,
    name="get_most_wickets", 
    description="Get the top wicket takers in the tournament"
)

most_boundaries_tool = StructuredTool.from_function(
    func=get_most_fours_and_sixes,
    name="get_most_fours_and_sixes",
    description="Get players with most fours and sixes"
)

# ------------------- Advanced Batting Tools -------------------------
runs_by_overs_tool = StructuredTool.from_function(
    func=lambda overs_start, overs_end, bowling_style: runs_by_overs_and_style(
        df, overs_start, overs_end, bowling_style
    ),
    name="runs_by_overs_and_style",
    description="Get batting performance in specific overs vs bowling style (e.g., pace, spin)"
)

death_overs_batting_tool = StructuredTool.from_function(
    func=lambda bowling_type: best_batters_death_overs(df, bowling_type),
    name="best_batters_death_overs",
    description="Get best batters in death overs (16-20) vs specific bowling type (pace/spin/all)"
)

batting_vs_hand_tool = StructuredTool.from_function(
    func=lambda bat_hand, overs_start, overs_end: best_batters_vs_batting_hand(
        df, bat_hand, overs_start, overs_end
    ),
    name="best_batters_vs_batting_hand",
    description="Get best batters vs specific batting hand (LHB/RHB) in given overs"
)

# ------------------- Advanced Bowling Tools -------------------------
bowler_runs_tool = StructuredTool.from_function(
    func=lambda overs_start, overs_end: bowler_conceded_runs(df, overs_start, overs_end),
    name="bowler_conceded_runs",
    description="Get runs conceded by bowlers in specific overs"
)

bowler_wickets_tool = StructuredTool.from_function(
    func=lambda overs_start, overs_end: bowler_wickets(df, overs_start, overs_end),
    name="bowler_wickets",
    description="Get wickets taken by bowlers in specific overs"
)

death_overs_bowling_tool = StructuredTool.from_function(
    func=lambda bat_hand: best_bowlers_death_overs(df, bat_hand),
    name="best_bowlers_death_overs",
    description="Get best bowlers in death overs (16-20) vs specific batting hand (LHB/RHB/all)"
)

bowling_vs_hand_tool = StructuredTool.from_function(
    func=lambda bat_hand, overs_start, overs_end: best_bowlers_vs_batting_hand(
        df, bat_hand, overs_start, overs_end
    ),
    name="best_bowlers_vs_batting_hand",
    description="Get best bowlers vs specific batting hand (LHB/RHB) in given overs"
)

bowler_economy_tool = StructuredTool.from_function(
    func=lambda overs_start, overs_end: bowler_economy(df, overs_start, overs_end),
    name="bowler_economy",
    description="Get economy rates for bowlers in specific overs"
)

# ------------------- Fielding Tools -------------------------
fielding_tool = StructuredTool.from_function(
    func=get_fielding_stats,
    name="get_fielding_stats",
    description="Get fielding statistics including catches, run-outs, stumpings"
)

# ------------------- Partnership Tools -------------------------
partnerships_tool = StructuredTool.from_function(
    func=get_best_partnerships,
    name="get_best_partnerships",
    description="Get the best batting partnerships in the tournament"
)

# ------------------- Team Analysis Tools -------------------------
team_runs_tool = StructuredTool.from_function(
    func=team_total_runs,
    name="team_total_runs",
    description="Get total runs scored by each team"
)

team_wickets_tool = StructuredTool.from_function(
    func=team_wickets_lost,
    name="team_wickets_lost",
    description="Get wickets lost by each team"
)

team_runrate_tool = StructuredTool.from_function(
    func=team_run_rate,
    name="team_run_rate",
    description="Get run rate for each team"
)

# ------------------- Match Analysis Tools -------------------------
match_results_tool = StructuredTool.from_function(
    func=get_match_results,
    name="get_match_results",
    description="Get match results and winners"
)

venue_stats_tool = StructuredTool.from_function(
    func=get_venue_stats,
    name="get_venue_stats",
    description="Get statistics by venue"
)

# ------------------- Advanced Analytics Tools -------------------------
powerplay_tool = StructuredTool.from_function(
    func=get_powerplay_stats,
    name="get_powerplay_stats",
    description="Get powerplay (1-6 overs) statistics for batting and bowling"
)

middle_overs_tool = StructuredTool.from_function(
    func=get_middle_overs_stats,
    name="get_middle_overs_stats",
    description="Get middle overs (7-15) statistics for batting and bowling"
)

death_overs_stats_tool = StructuredTool.from_function(
    func=get_death_overs_stats,
    name="get_death_overs_stats",
    description="Get death overs (16-20) statistics for batting and bowling"
)

# ------------------- Team-Specific Tools -------------------------
team_batters_tool = StructuredTool.from_function(
    func=lambda team_name: get_team_best_batters(df, team_name),
    name="get_team_best_batters",
    description="Get the best batters for a specific team (e.g., Chennai Super Kings, Mumbai Indians)"
)

team_bowlers_tool = StructuredTool.from_function(
    func=lambda team_name: get_team_best_bowlers(df, team_name),
    name="get_team_best_bowlers",
    description="Get the best bowlers for a specific team"
)

team_stats_tool = StructuredTool.from_function(
    func=lambda team_name: get_team_overall_stats(df, team_name),
    name="get_team_overall_stats",
    description="Get overall statistics for a specific team"
)

# ------------------- Utility Tools -------------------------
strike_rate_tool = StructuredTool.from_function(
    func=lambda min_balls: top_strike_rate(df, min_balls),
    name="top_strike_rate",
    description="Get top strike rates for batters with minimum balls faced"
)

win_prob_tool = StructuredTool.from_function(
    func=lambda team: win_probability(df, team),
    name="win_probability",
    description="Get win probability for a specific team"
)

# ------------------- Add all tools to the list -------------------------
tools.extend([
    most_runs_tool,
    most_wickets_tool,
    most_boundaries_tool,
    runs_by_overs_tool,
    death_overs_batting_tool,
    batting_vs_hand_tool,
    bowler_runs_tool,
    bowler_wickets_tool,
    death_overs_bowling_tool,
    bowling_vs_hand_tool,
    bowler_economy_tool,
    fielding_tool,
    partnerships_tool,
    team_runs_tool,
    team_wickets_tool,
    team_runrate_tool,
    match_results_tool,
    venue_stats_tool,
    powerplay_tool,
    middle_overs_tool,
    death_overs_stats_tool,
    team_batters_tool,
    team_bowlers_tool,
    team_stats_tool,
    strike_rate_tool,
    win_prob_tool,
])

# -------------------------------------------------
# 3️⃣  Create the agent
# -------------------------------------------------
# Initialize the agent with our custom tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

# Set a better system prompt for the agent
if hasattr(agent, 'agent') and hasattr(agent.agent, 'llm_chain'):
    agent.agent.llm_chain.prompt.template = """You are a cricket analytics expert assistant. You have access to comprehensive IPL 2025 cricket data and specialized tools to analyze it.

When users ask questions about cricket, use the appropriate tools to provide detailed, accurate answers. Always use the specific tools available rather than generic responses.

Available teams in the dataset:
- Chennai Super Kings
- Mumbai Indians
- Royal Challengers Bengaluru
- Kolkata Knight Riders
- Delhi Capitals
- Punjab Kings
- Rajasthan Royals
- Gujarat Titans
- Lucknow Super Giants
- Sunrisers Hyderabad

For team-specific queries, use the team tools like get_team_best_batters, get_team_best_bowlers, or get_team_overall_stats.

Always provide detailed, formatted responses with tables and statistics when available.

{input}
{agent_scratchpad}"""

# -------------------------------------------------
# 4️⃣  Streamlit UI
# -------------------------------------------------
st.set_page_config(
    page_title="🏏 IPL 2025 Cricket Analytics Chatbot",
    page_icon="🏏",
    layout="wide",
)

st.title("🏏 IPL 2025 Cricket Analytics Chatbot")
st.markdown("""
This chatbot can answer comprehensive cricket queries about IPL 2025 including:

### 🎯 **Simple Queries**
- Most runs, wickets, fours, sixes
- Team statistics and rankings
- Match results and venues

### 🚀 **Advanced Batting Analysis**
- Best batters in death overs vs pace/spin
- Performance vs specific bowling types
- Strike rates and boundary analysis

### 🎾 **Advanced Bowling Analysis**  
- Best bowlers in death overs vs RHB/LHB
- Economy rates in specific overs
- Wicket-taking analysis

### 🏃‍♂️ **Fielding & Partnerships**
- Catches, run-outs, stumpings
- Best batting partnerships
- Team fielding performance

### 📊 **Phase-wise Analysis**
- Powerplay (1-6 overs) statistics
- Middle overs (7-15) performance
- Death overs (16-20) analysis

**Ask me anything about IPL 2025 cricket!** 🏏
""")

# -------------------------------------------------
# 5️⃣  Chat interface
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about IPL 2025 cricket..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Create callback handler for streaming
        callback_handler = StreamlitCallbackHandler(
            parent_container=message_placeholder,
            max_thought_containers=10,
            expand_new_thoughts=True,
            collapse_completed_thoughts=True,
        )
        
        try:
            # Run the agent with streaming
            response = agent.run(
                prompt,
                callbacks=[callback_handler],
            )
            full_response = response
            
        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            st.error(f"An error occurred: {str(e)}")
        
        # Update the message placeholder with final response
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# -------------------------------------------------
# 6️⃣  Sidebar with example queries
# -------------------------------------------------
with st.sidebar:
    st.header("💡 Example Queries")
    
    st.subheader("🎯 Basic Stats")
    if st.button("Most runs in IPL 2025"):
        st.session_state.messages.append({"role": "user", "content": "Who has scored the most runs in IPL 2025?"})
        st.rerun()
    
    if st.button("Most wickets"):
        st.session_state.messages.append({"role": "user", "content": "Who has taken the most wickets in IPL 2025?"})
        st.rerun()
    
    if st.button("Most boundaries"):
        st.session_state.messages.append({"role": "user", "content": "Who has hit the most fours and sixes?"})
        st.rerun()
    
    st.subheader("🚀 Advanced Batting")
    if st.button("Best batters in death overs"):
        st.session_state.messages.append({"role": "user", "content": "Who are the best batters in death overs (16-20) vs pace bowling?"})
        st.rerun()
    
    if st.button("Best batters vs RHB"):
        st.session_state.messages.append({"role": "user", "content": "Who are the best batters vs right-handed batsmen in overs 16-20?"})
        st.rerun()
    
    st.subheader("🎾 Advanced Bowling")
    if st.button("Best bowlers in death overs"):
        st.session_state.messages.append({"role": "user", "content": "Who are the best bowlers in death overs vs right-handed batsmen?"})
        st.rerun()
    
    if st.button("Best economy rates"):
        st.session_state.messages.append({"role": "user", "content": "Who has the best economy rate in overs 16-20?"})
        st.rerun()
    
    st.subheader("📊 Team Analysis")
    if st.button("Team run rates"):
        st.session_state.messages.append({"role": "user", "content": "What are the team run rates in IPL 2025?"})
        st.rerun()
    
    if st.button("Powerplay stats"):
        st.session_state.messages.append({"role": "user", "content": "Show me powerplay statistics for all teams"})
        st.rerun()
    
    st.subheader("🏃‍♂️ Fielding & Partnerships")
    if st.button("Fielding stats"):
        st.session_state.messages.append({"role": "user", "content": "Show me the best fielders with most catches and run-outs"})
        st.rerun()
    
    if st.button("Best partnerships"):
        st.session_state.messages.append({"role": "user", "content": "What are the best batting partnerships in IPL 2025?"})
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# -------------------------------------------------
# 7️⃣  Footer
# -------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏏 IPL 2025 Cricket Analytics Chatbot | Powered by LangChain & Groq 🚀</p>
    <p>Ask me anything about batting, bowling, fielding, partnerships, and team performance!</p>
    <p style='font-size: 12px; color: #999;'>💰 Ultra-fast inference • FREE tier available • 70% cheaper than alternatives</p>
</div>
""", unsafe_allow_html=True)
