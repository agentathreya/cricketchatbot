"""
app.py
-------

Streamlit UI + LangChain cricket analytics chatbot using Groq API.
Optimized for deployment with proper error handling and Groq integration.
"""

import os
import sys
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import traceback

# LangChain imports
try:
    from langchain.agents import AgentType, initialize_agent
    from langchain.tools import StructuredTool
    from langchain_groq import ChatGroq
    from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
except ImportError as e:
    st.error(f"Missing required packages. Please run: pip install -r requirements.txt")
    st.error(f"Import error: {e}")
    st.stop()

# Load environment variables
load_dotenv()

# ===================================================================
# 🔑 API KEY SETUP WITH PROPER ERROR HANDLING
# ===================================================================

def setup_groq_api():
    """Setup Groq API with proper error handling"""
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY not found!")
        st.info("""
        **Setup Instructions:**
        
        1. Get FREE API key: https://console.groq.com/
        2. For local development:
           ```
           echo "GROQ_API_KEY=your_key_here" > .env
           ```
        3. For deployment, add GROQ_API_KEY to your platform's environment variables
        
        **Why Groq?**
        - 🆓 FREE tier: 14,400 requests/day
        - ⚡ Ultra-fast responses (fastest in market)
        - 💰 70% cheaper than alternatives
        """)
        return None
    
    return groq_api_key

# Setup API
api_key = setup_groq_api()
if not api_key:
    st.stop()

st.success("✅ Groq API configured successfully! 🚀")

# ===================================================================
# 📊 DATA LOADING WITH ERROR HANDLING
# ===================================================================

@st.cache_data(ttl=7200)  # Cache for 2 hours
def load_cricket_data():
    """Load and clean cricket data with error handling"""
    try:
        # Try different possible paths for the CSV file
        possible_paths = [
            "cricket_data.csv",
            "./cricket_data.csv",
            os.path.join(os.getcwd(), "cricket_data.csv")
        ]
        
        df = None
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path, low_memory=False)
                    st.success(f"📊 Data loaded from: {path}")
                    break
            except Exception as e:
                continue
        
        if df is None:
            st.error("❌ Could not find cricket_data.csv file!")
            st.info("Please ensure cricket_data.csv is in the project root directory.")
            return None
        
        # Data cleaning
        numeric_cols = ["over", "runs_total", "runs_batter", "balls_faced"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Boolean conversion
        bool_cols = ["isFour", "isSix", "isWicket"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # String columns
        str_cols = ["bowling_style", "bowling_type", "bat_hand"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
df = load_cricket_data()
if df is None:
    st.stop()

st.info(f"📈 Dataset: {len(df):,} records loaded successfully")

# ===================================================================
# 🤖 LLM SETUP WITH GROQ
# ===================================================================

def create_groq_llm():
    """Create Groq LLM with error handling"""
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-70b-8192",  # Best model for cricket analysis
            temperature=0,
            max_retries=3,
            timeout=60,  # Increased timeout for complex queries
            streaming=True
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Groq LLM: {str(e)}")
        return None

llm = create_groq_llm()
if llm is None:
    st.stop()

# ===================================================================
# 🛠️ CRICKET ANALYSIS FUNCTIONS
# ===================================================================

def safe_function_call(func, *args, **kwargs):
    """Wrapper for safe function calls with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return f"Error in analysis: {str(e)}"

# Import utility functions with error handling
try:
    from utils import (
        get_most_runs, get_most_wickets, get_most_fours_and_sixes,
        runs_by_overs_and_style, best_batters_death_overs, best_batters_vs_batting_hand,
        bowler_conceded_runs, bowler_wickets, best_bowlers_death_overs,
        best_bowlers_vs_batting_hand, bowler_economy, get_fielding_stats,
        get_best_partnerships, team_total_runs, team_wickets_lost, team_run_rate,
        get_match_results, get_venue_stats, get_powerplay_stats,
        get_middle_overs_stats, get_death_overs_stats, get_team_best_batters,
        get_team_best_bowlers, get_team_overall_stats, top_strike_rate,
        win_probability, best_partnerships
    )
    st.success("✅ All cricket analysis functions loaded successfully")
except ImportError as e:
    st.error(f"Error importing utils: {e}")
    st.stop()

# ===================================================================
# 🔧 TOOL CREATION
# ===================================================================

def create_tools():
    """Create LangChain tools with error handling"""
    tools = []
    
    # Basic statistics tools
    basic_tools = [
        ("get_most_runs", lambda: safe_function_call(get_most_runs, df), "Get top run scorers"),
        ("get_most_wickets", lambda: safe_function_call(get_most_wickets, df), "Get top wicket takers"),
        ("get_most_boundaries", lambda: safe_function_call(get_most_fours_and_sixes, df), "Get players with most fours and sixes"),
    ]
    
    # Team analysis tools
    team_tools = [
        ("team_total_runs", lambda: safe_function_call(team_total_runs, df), "Get total runs by team"),
        ("team_run_rate", lambda: safe_function_call(team_run_rate, df), "Get run rates by team"),
        ("powerplay_stats", lambda: safe_function_call(get_powerplay_stats, df), "Get powerplay statistics"),
        ("middle_overs_stats", lambda: safe_function_call(get_middle_overs_stats, df), "Get middle overs statistics"),
        ("death_overs_stats", lambda: safe_function_call(get_death_overs_stats, df), "Get death overs statistics"),
    ]
    
    # Advanced analysis tools
    advanced_tools = [
        ("fielding_stats", lambda: safe_function_call(get_fielding_stats, df), "Get fielding statistics"),
        ("best_partnerships", lambda: safe_function_call(get_best_partnerships, df), "Get best partnerships"),
        ("venue_stats", lambda: safe_function_call(get_venue_stats, df), "Get venue statistics"),
    ]
    
    # Create StructuredTool objects
    all_tools = basic_tools + team_tools + advanced_tools
    
    for name, func, description in all_tools:
        try:
            tool = StructuredTool.from_function(
                func=func,
                name=name,
                description=description
            )
            tools.append(tool)
        except Exception as e:
            st.warning(f"Could not create tool {name}: {e}")
            continue
    
    return tools

tools = create_tools()
st.success(f"✅ Created {len(tools)} analysis tools")

# ===================================================================
# 🤖 AGENT CREATION
# ===================================================================

def create_agent():
    """Create LangChain agent with error handling"""
    try:
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,  # Reduced for faster responses
            early_stopping_method="generate"
        )
        return agent
    except Exception as e:
        st.error(f"Failed to create agent: {str(e)}")
        return None

agent = create_agent()
if agent is None:
    st.stop()

# ===================================================================
# 🎨 STREAMLIT UI
# ===================================================================

st.set_page_config(
    page_title="🏏 IPL Cricket Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏏 IPL 2025 Cricket Analytics Chatbot")
st.markdown("""
**Powered by Groq API** - Ultra-fast cricket insights with AI!

### 🎯 What I can analyze:
- **Basic Stats**: Top run scorers, wicket takers, boundaries
- **Team Performance**: Run rates, powerplay stats, team rankings  
- **Advanced Analytics**: Death overs analysis, partnerships, venue stats
- **Player Insights**: Strike rates, economy rates, fielding stats

**Ask me anything about IPL cricket!** ⚡
""")

# ===================================================================
# 💬 CHAT INTERFACE
# ===================================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your IPL cricket analytics assistant. Ask me anything about player performance, team stats, or match insights! 🏏"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about IPL cricket stats..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing cricket data..."):
            try:
                # Create callback handler
                callback_handler = StreamlitCallbackHandler(st.container())
                
                # Get response from agent
                response = agent.run(
                    input=prompt,
                    callbacks=[callback_handler]
                )
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ===================================================================
# 📋 SIDEBAR WITH EXAMPLES
# ===================================================================

with st.sidebar:
    st.header("💡 Example Queries")
    
    st.subheader("🎯 Basic Stats")
    if st.button("Top run scorers", key="runs"):
        st.session_state.messages.append({"role": "user", "content": "Who are the top run scorers in IPL 2025?"})
        st.rerun()
    
    if st.button("Leading wicket takers", key="wickets"):
        st.session_state.messages.append({"role": "user", "content": "Show me the leading wicket takers"})
        st.rerun()
    
    st.subheader("📊 Team Analysis")
    if st.button("Team run rates", key="team_rr"):
        st.session_state.messages.append({"role": "user", "content": "What are the team run rates?"})
        st.rerun()
    
    if st.button("Powerplay performance", key="powerplay"):
        st.session_state.messages.append({"role": "user", "content": "Show powerplay statistics for all teams"})
        st.rerun()
    
    st.subheader("⚡ Advanced")
    if st.button("Death overs analysis", key="death"):
        st.session_state.messages.append({"role": "user", "content": "Analyze death overs performance"})
        st.rerun()
    
    if st.button("Best partnerships", key="partnerships"):
        st.session_state.messages.append({"role": "user", "content": "What are the best batting partnerships?"})
        st.rerun()
    
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared! Ask me anything about IPL cricket! 🏏"}
        ]
        st.rerun()

# ===================================================================
# 📊 FOOTER
# ===================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏏 <b>IPL Cricket Analytics Chatbot</b> | Powered by <b>Groq API</b> ⚡</p>
    <p style='font-size: 12px;'>Ultra-fast inference • FREE tier • 70% cheaper than alternatives</p>
</div>
""", unsafe_allow_html=True)
