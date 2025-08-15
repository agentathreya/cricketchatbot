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
            timeout=90,  # Increased timeout for step-by-step reasoning
            streaming=True
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Groq LLM: {str(e)}")
        return None

def create_reasoning_llm():
    """Create a separate LLM instance for reasoning and planning"""
    try:
        reasoning_llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-70b-8192", 
            temperature=0.1,  # Slightly higher for creative reasoning
            max_retries=2,
            timeout=30
        )
        return reasoning_llm
    except Exception as e:
        st.error(f"Failed to initialize reasoning LLM: {str(e)}")
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
# 🔧 TOOL CREATION - Fixed for Agent Compatibility
# ===================================================================

def create_wrapper_functions():
    """Create wrapper functions that work with LangChain agents"""
    
    def get_runs_wrapper():
        """Get top run scorers"""
        return safe_function_call(get_most_runs, df)
    
    def get_wickets_wrapper():
        """Get top wicket takers"""
        return safe_function_call(get_most_wickets, df)
    
    def get_boundaries_wrapper():
        """Get players with most fours and sixes"""
        return safe_function_call(get_most_fours_and_sixes, df)
    
    def get_team_runs_wrapper():
        """Get total runs by team"""
        return safe_function_call(team_total_runs, df)
    
    def get_team_rates_wrapper():
        """Get run rates by team"""
        return safe_function_call(team_run_rate, df)
    
    def get_powerplay_wrapper():
        """Get powerplay statistics"""
        return safe_function_call(get_powerplay_stats, df)
    
    def get_middle_overs_wrapper():
        """Get middle overs statistics"""
        return safe_function_call(get_middle_overs_stats, df)
    
    def get_death_overs_wrapper():
        """Get death overs statistics"""
        return safe_function_call(get_death_overs_stats, df)
    
    def get_fielding_wrapper():
        """Get fielding statistics"""
        return safe_function_call(get_fielding_stats, df)
    
    def get_partnerships_wrapper():
        """Get best partnerships"""
        return safe_function_call(get_best_partnerships, df)
    
    def get_venues_wrapper():
        """Get venue statistics"""
        return safe_function_call(get_venue_stats, df)
    
    return {
        "get_most_runs": get_runs_wrapper,
        "get_most_wickets": get_wickets_wrapper, 
        "get_most_boundaries": get_boundaries_wrapper,
        "team_total_runs": get_team_runs_wrapper,
        "team_run_rates": get_team_rates_wrapper,
        "powerplay_stats": get_powerplay_wrapper,
        "middle_overs_stats": get_middle_overs_wrapper,
        "death_overs_stats": get_death_overs_wrapper,
        "fielding_stats": get_fielding_wrapper,
        "best_partnerships": get_partnerships_wrapper,
        "venue_stats": get_venues_wrapper
    }

def create_tools():
    """Create LangChain tools with proper function wrapping"""
    wrapper_functions = create_wrapper_functions()
    tools = []
    
    tool_definitions = [
        ("get_most_runs", "Get the top run scorers in IPL 2025"),
        ("get_most_wickets", "Get the top wicket takers in IPL 2025"),
        ("get_most_boundaries", "Get players with most fours and sixes"),
        ("team_total_runs", "Get total runs scored by each team"),
        ("team_run_rates", "Get run rates for each team"),
        ("powerplay_stats", "Get powerplay (overs 1-6) statistics"),
        ("middle_overs_stats", "Get middle overs (7-15) statistics"), 
        ("death_overs_stats", "Get death overs (16-20) statistics"),
        ("fielding_stats", "Get fielding statistics including catches and run-outs"),
        ("best_partnerships", "Get the best batting partnerships"),
        ("venue_stats", "Get statistics by venue")
    ]
    
    for name, description in tool_definitions:
        try:
            if name in wrapper_functions:
                tool = StructuredTool.from_function(
                    func=wrapper_functions[name],
                    name=name,
                    description=description
                )
                tools.append(tool)
                print(f"✅ Created tool: {name}")
        except Exception as e:
            print(f"⚠️  Could not create tool {name}: {e}")
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
        # Try different agent types for better compatibility
        agent_types = [
            AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION
        ]
        
        for agent_type in agent_types:
            try:
                agent = initialize_agent(
                    tools=tools,
                    llm=llm,
                    agent=agent_type,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=5,
                    early_stopping_method="generate"
                )
                st.success(f"✅ Created agent with type: {agent_type.value}")
                return agent
            except Exception as e:
                st.warning(f"Agent type {agent_type.value} failed: {str(e)}")
                continue
        
        raise Exception("All agent types failed to initialize")
        
    except Exception as e:
        st.error(f"Failed to create agent: {str(e)}")
        st.info("""**Troubleshooting:** 
        1. Check if all required packages are installed
        2. Ensure cricket data is loaded properly
        3. Verify Groq API key is valid
        """)
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
# 📚 DATASET KNOWLEDGE SYSTEM
# ===================================================================

# Comprehensive understanding of the IPL 2025 dataset based on analysis
DATASET_KNOWLEDGE = {
    "structure": {
        "total_records": "17,176 ball-by-ball records",
        "match_format": "T20 format (20 overs per innings)",
        "season": "IPL 2025",
        "data_type": "Ball-by-ball detailed match data"
    },
    "teams": [
        "Kolkata Knight Riders", "Royal Challengers Bengaluru", 
        "Chennai Super Kings", "Mumbai Indians", "Delhi Capitals",
        "Sunrisers Hyderabad", "Rajasthan Royals", "Punjab Kings",
        "Lucknow Super Giants", "Gujarat Titans"
    ],
    "key_players": {
        "batters": ["V Kohli", "PD Salt", "AM Rahane", "SP Narine", "Q de Kock", "VR Iyer", 
                   "A Raghuvanshi", "AD Russell", "Ramandeep Singh", "RK Singh"],
        "bowlers": ["JR Hazlewood", "Yash Dayal", "KH Pandya", "Rasikh Salam", "Suyash Sharma",
                   "CV Varun", "SH Johnson", "Harshit Rana", "LS Livingstone", "VG Arora"]
    },
    "venues": ["Eden Gardens, Kolkata"],  # From the sample data
    "statistics_available": {
        "batting": ["runs_total", "runs_batter", "balls_faced", "isFour", "isSix", "curr_batter_runs", "curr_batter_balls"],
        "bowling": ["bowler_runs", "bowler_wicket", "bowling_style", "bowling_type", "isWicket", "dismissal_type"],
        "team": ["team_runs", "team_balls", "team_wickets", "Current RR", "Required RR", "winProbabilty"],
        "match": ["over", "ball", "innings", "predictedScore", "batting_partners", "venue", "date"],
        "detailed": ["wagonX", "wagonY", "wagonZone", "pitchLine", "pitchLength", "shotType", "shotControl"]
    },
    "match_phases": {
        "powerplay": "Overs 1-6",
        "middle_overs": "Overs 7-15", 
        "death_overs": "Overs 16-20"
    },
    "shot_types": ["CUT_SHOT", "PULL", "FLICK", "ON_DRIVE", "COVER_DRIVE", "STRAIGHT_DRIVE", 
                   "SLOG_SHOT", "SWEEP_SHOT", "REVERSE_SWEEP", "SQUARE_DRIVE", "LEG_GLANCE"],
    "bowling_styles": ["rfm", "lmf", "rm", "sla", "lbg", "lf", "rf", "ob"],
    "dismissal_types": ["caught", "bowled"],
    "pitch_locations": ["ON_THE_STUMPS", "OUTSIDE_OFFSTUMP", "DOWN_LEG", "WIDE_OUTSIDE_OFFSTUMP"],
    "ball_lengths": ["FULL", "GOOD_LENGTH", "SHORT_OF_A_GOOD_LENGTH", "SHORT", "YORKER", "FULL_TOSS"]
}

def get_dataset_context():
    """Generate context about what's available in the dataset"""
    return f"""
    📊 **IPL 2025 Dataset Overview:**
    - **Records**: {DATASET_KNOWLEDGE['structure']['total_records']}
    - **Teams**: {len(DATASET_KNOWLEDGE['teams'])} IPL teams
    - **Format**: {DATASET_KNOWLEDGE['structure']['match_format']}
    - **Data Type**: {DATASET_KNOWLEDGE['structure']['data_type']}
    
    🏏 **Available Statistics:**
    • Batting: Runs, balls faced, boundaries (4s/6s), strike rates
    • Bowling: Wickets, economy rates, bowling styles, dismissals
    • Team: Run rates, win probabilities, partnerships
    • Match Phases: Powerplay, middle overs, death overs analysis
    • Advanced: Shot analysis, pitch maps, wagon wheels
    
    🎯 **Key Features:**
    • Ball-by-ball tracking with exact over/ball details
    • Shot type and control analysis
    • Bowling line/length data
    • Real-time win probability tracking
    • Partnership and batting position tracking
    """

def validate_query_against_dataset(query: str) -> dict:
    """Check if a query can be answered with available data"""
    query_lower = query.lower()
    
    # Check for team names
    mentioned_teams = [team for team in DATASET_KNOWLEDGE['teams'] if team.lower() in query_lower]
    
    # Check for player names
    mentioned_players = []
    for category in DATASET_KNOWLEDGE['key_players'].values():
        mentioned_players.extend([player for player in category if player.lower() in query_lower])
    
    # Check for statistical categories
    stat_categories = []
    if any(word in query_lower for word in ['run', 'score', 'batting']):
        stat_categories.append('batting')
    if any(word in query_lower for word in ['wicket', 'bowl', 'economy']):
        stat_categories.append('bowling')
    if any(word in query_lower for word in ['team', 'total']):
        stat_categories.append('team')
    if any(word in query_lower for word in ['powerplay', 'death', 'middle']):
        stat_categories.append('match_phases')
    
    return {
        'teams': mentioned_teams,
        'players': mentioned_players,
        'stat_categories': stat_categories,
        'is_answerable': len(mentioned_teams) > 0 or len(mentioned_players) > 0 or len(stat_categories) > 0
    }

# ===================================================================
# 🧠 ENHANCED STRUCTURED REASONING SYSTEM
# ===================================================================

def analyze_query_step_by_step(user_query: str):
    """Implement structured reasoning: Understand → Plan → Execute → Present"""
    
    # Create reasoning LLM
    reasoning_llm = create_reasoning_llm()
    if not reasoning_llm:
        return None, "Failed to initialize reasoning system"
    
    # Pre-analysis: Validate query against dataset
    query_validation = validate_query_against_dataset(user_query)
    dataset_context = get_dataset_context()
    
    # Step 1: 🎯 UNDERSTAND the query (Dataset-aware)
    with st.expander("🎯 **Step 1: Understanding Your Question**", expanded=True):
        st.write("Analyzing what you want to know using dataset knowledge...")
        
        # Show validation results
        if query_validation['teams']:
            st.success(f"🏆 Teams found: {', '.join(query_validation['teams'])}")
        if query_validation['players']:
            st.success(f"👨‍💻 Players found: {', '.join(query_validation['players'])}")
        if query_validation['stat_categories']:
            st.success(f"📊 Stats categories: {', '.join(query_validation['stat_categories'])}")
        
        if not query_validation['is_answerable']:
            st.warning("⚠️ Query might not match available data. Suggesting alternatives...")
        
        understand_prompt = f"""
        You are a cricket analytics expert with deep knowledge of the IPL 2025 dataset.
        
        DATASET CONTEXT:
        {dataset_context}
        
        QUERY VALIDATION:
        - Teams mentioned: {query_validation['teams']}
        - Players mentioned: {query_validation['players']}
        - Statistical categories: {query_validation['stat_categories']}
        - Can be answered: {query_validation['is_answerable']}
        
        Analyze this user question:
        "{user_query}"
        
        Respond in this format:
        **Query Type:** [batting/bowling/team/fielding/partnership/venue analysis]
        **Entities Found:** [specific players/teams identified from dataset]
        **Available Data:** [confirm what data we have to answer this]
        **Analysis Level:** [basic stats/advanced analysis/comparison]
        **Intent:** [brief summary of what user wants to know]
        **Feasibility:** [HIGH/MEDIUM/LOW - how well we can answer this]
        """
        
        try:
            understanding = reasoning_llm.invoke(understand_prompt).content
            st.markdown(understanding)
        except Exception as e:
            understanding = f"Error in understanding: {str(e)}"
            st.error(understanding)
    
    # Step 2: 📋 PLAN the analysis
    with st.expander("📋 **Step 2: Planning the Analysis**", expanded=True):
        st.write("Determining the best approach...")
        
        # Available tools for planning
        available_tools = [
            "get_most_runs - Top run scorers",
            "get_most_wickets - Top wicket takers", 
            "get_most_boundaries - Most fours and sixes",
            "team_total_runs - Team run totals",
            "team_run_rates - Team run rates",
            "powerplay_stats - Powerplay statistics",
            "middle_overs_stats - Middle overs stats",
            "death_overs_stats - Death overs stats",
            "fielding_stats - Fielding statistics",
            "best_partnerships - Best partnerships",
            "venue_stats - Venue statistics"
        ]
        
        plan_prompt = f"""
        Based on the user query and understanding, determine the best analysis approach.
        
        User Query: "{user_query}"
        Understanding: {understanding}
        
        Available Analysis Tools:
        {chr(10).join(available_tools)}
        
        Respond with:
        **Selected Tool:** [exact tool name from the list above]
        **Reasoning:** [why this tool is best for the query]
        **Expected Output:** [what kind of results this will provide]
        """
        
        try:
            plan = reasoning_llm.invoke(plan_prompt).content
            st.markdown(plan)
            
            # Extract the selected tool
            selected_tool = None
            for line in plan.split('\n'):
                if line.startswith('**Selected Tool:**'):
                    tool_name = line.split(':', 1)[1].strip()
                    selected_tool = tool_name
                    break
        except Exception as e:
            plan = f"Error in planning: {str(e)}"
            st.error(plan)
            selected_tool = None
    
    # Step 3: ⚡ EXECUTE the analysis
    with st.expander("⚡ **Step 3: Executing the Analysis**", expanded=True):
        st.write("Running the cricket data analysis...")
        
        if selected_tool:
            try:
                # Map tool names to functions
                tool_mapping = {
                    "get_most_runs": get_most_runs,
                    "get_most_wickets": get_most_wickets,
                    "get_most_boundaries": get_most_fours_and_sixes,
                    "team_total_runs": team_total_runs,
                    "team_run_rates": team_run_rate,
                    "powerplay_stats": get_powerplay_stats,
                    "middle_overs_stats": get_middle_overs_stats,
                    "death_overs_stats": get_death_overs_stats,
                    "fielding_stats": get_fielding_stats,
                    "best_partnerships": get_best_partnerships,
                    "venue_stats": get_venue_stats
                }
                
                if selected_tool in tool_mapping:
                    with st.spinner(f"Running {selected_tool} analysis..."):
                        result = safe_function_call(tool_mapping[selected_tool], df)
                    st.success("✅ Analysis completed!")
                    st.code(f"Executed: {selected_tool}(cricket_data)", language="python")
                else:
                    result = "Tool not found. Using default analysis..."
                    result = safe_function_call(get_most_runs, df)
            except Exception as e:
                result = f"Error executing analysis: {str(e)}"
                st.error(result)
        else:
            result = "Could not determine appropriate analysis tool"
            st.warning(result)
    
    # Step 4: 📊 PRESENT the results  
    with st.expander("📊 **Step 4: Presenting the Results**", expanded=True):
        st.write("Formatting and presenting your cricket insights...")
        
        if result and not result.startswith("Error"):
            # Generate a comprehensive explanation
            present_prompt = f"""
            You are a cricket commentator explaining analysis results to fans.
            
            Original Question: "{user_query}"
            Analysis Results: {result}
            
            Create a comprehensive response that:
            1. Directly answers the user's question
            2. Provides context and insights from the data
            3. Highlights key findings and interesting patterns
            4. Uses cricket terminology appropriately
            
            Format your response with clear headings and bullet points for readability.
            """
            
            try:
                final_response = reasoning_llm.invoke(present_prompt).content
                st.markdown("### 🏏 **Cricket Analysis Results**")
                st.markdown(final_response)
                
                # Also show the raw data in an expandable section
                with st.expander("📈 **Raw Data Results**"):
                    if isinstance(result, str) and '|' in result:
                        st.markdown(result)
                    else:
                        st.text(result)
                        
                return final_response, None
            except Exception as e:
                error_msg = f"Error in presentation: {str(e)}"
                st.error(error_msg)
                return result, error_msg
        else:
            st.error("No valid results to present")
            return result, "No valid results"

# ===================================================================
# 💬 CHAT INTERFACE WITH STRUCTURED REASONING
# ===================================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your IPL cricket analytics assistant. Ask me anything about player performance, team stats, or match insights! 🏏"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Analysis mode toggle
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    analysis_mode = st.radio(
        "🔬 Analysis Mode:",
        ["🤖 Standard Chat", "🧠 Step-by-Step Reasoning"],
        index=0
    )

with col2:
    st.markdown("**Analysis Mode Info:**")
    if "Standard" in analysis_mode:
        st.info("💬 Direct LLM agent responses")
    else:
        st.info("🔍 Structured reasoning with 4 steps")

# Chat input
if prompt := st.chat_input("Ask about IPL cricket stats..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response based on selected mode
    with st.chat_message("assistant"):
        if "Step-by-Step" in analysis_mode:
            # Use structured reasoning approach
            st.markdown("## 🧠 **Step-by-Step Cricket Analysis**")
            response, error = analyze_query_step_by_step(prompt)
            if error:
                st.error(f"Analysis error: {error}")
            
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            # Use enhanced standard agent approach with dataset awareness
            query_validation = validate_query_against_dataset(prompt)
            
            # Show quick validation results
            if query_validation['teams'] or query_validation['players']:
                entities = query_validation['teams'] + query_validation['players']
                st.info(f"🎯 Found entities: {', '.join(entities[:3])}{'...' if len(entities) > 3 else ''}")
            
            if not query_validation['is_answerable']:
                st.warning("⚠️ This query might not match our dataset. I'll do my best to provide relevant information.")
            
            with st.spinner("Analyzing cricket data with dataset intelligence..."):
                try:
                    # Enhance the prompt with dataset context for better responses
                    enhanced_prompt = f"""
                    Dataset Context: {get_dataset_context()}
                    
                    User Query: {prompt}
                    
                    Please provide a comprehensive answer based on the available IPL 2025 data.
                    """
                    
                    # Create callback handler
                    callback_handler = StreamlitCallbackHandler(st.container())
                    
                    # Get response from agent
                    response = agent.run(
                        input=enhanced_prompt,
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
    st.header("📊 Dataset Overview")
    st.info(f"""
    **IPL 2025 Data**
    • {len(df):,} ball-by-ball records
    • {len(DATASET_KNOWLEDGE['teams'])} teams
    • Ball-by-ball match data
    • Shot analysis & wagon wheels
    • Real-time win probabilities
    """)
    
    # Smart suggestions based on dataset
    with st.expander("🏏 Teams Available"):
        teams_text = "\n".join([f"• {team}" for team in DATASET_KNOWLEDGE['teams'][:5]])
        st.text(teams_text)
        if len(DATASET_KNOWLEDGE['teams']) > 5:
            st.text(f"...and {len(DATASET_KNOWLEDGE['teams']) - 5} more")
    
    with st.expander("👥 Key Players"):
        st.text("**Batters:**")
        batters_text = "\n".join([f"• {player}" for player in DATASET_KNOWLEDGE['key_players']['batters'][:5]])
        st.text(batters_text)
        st.text("\n**Bowlers:**")
        bowlers_text = "\n".join([f"• {player}" for player in DATASET_KNOWLEDGE['key_players']['bowlers'][:5]])
        st.text(bowlers_text)
    
    st.divider()
    st.header("💡 Smart Query Suggestions")
    
    st.subheader("🎯 Player Analysis")
    if st.button("V Kohli's performance", key="kohli"):
        st.session_state.messages.append({"role": "user", "content": "How did V Kohli perform this season?"})
        st.rerun()
    
    if st.button("SP Narine batting stats", key="narine"):
        st.session_state.messages.append({"role": "user", "content": "Show SP Narine's batting performance"})
        st.rerun()
    
    st.subheader("🏆 Team Comparisons")
    if st.button("KKR vs RCB head-to-head", key="teams"):
        st.session_state.messages.append({"role": "user", "content": "Compare Kolkata Knight Riders vs Royal Challengers Bengaluru performance"})
        st.rerun()
    
    if st.button("Best team in powerplay", key="team_pp"):
        st.session_state.messages.append({"role": "user", "content": "Which team performed best in powerplay overs?"})
        st.rerun()
    
    st.subheader("📈 Advanced Analytics")
    if st.button("Shot analysis patterns", key="shots"):
        st.session_state.messages.append({"role": "user", "content": "Analyze shot patterns and types in IPL 2025"})
        st.rerun()
    
    if st.button("Win probability insights", key="win_prob"):
        st.session_state.messages.append({"role": "user", "content": "Show me interesting win probability moments"})
        st.rerun()
    
    st.subheader("🎯 Quick Stats")
    if st.button("Top boundaries", key="boundaries"):
        st.session_state.messages.append({"role": "user", "content": "Who hit the most boundaries (4s and 6s)?"})
        st.rerun()
    
    if st.button("Economy rates", key="economy"):
        st.session_state.messages.append({"role": "user", "content": "Show bowlers with best economy rates"})
        st.rerun()
    
    st.divider()
    st.subheader("🔧 Chat Controls")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared! Ask me anything about IPL cricket! 🏏"}
        ]
        st.rerun()
    
    # Query tips
    with st.expander("💡 Query Tips"):
        st.markdown("""
        **Try asking about:**
        • Specific player stats
        • Team comparisons
        • Match phase analysis
        • Shot type patterns
        • Bowling styles
        • Partnership details
        • Venue statistics
        """)

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
