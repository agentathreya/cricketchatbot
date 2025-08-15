# 🏏 IPL 2025 Cricket Analytics Chatbot

A comprehensive cricket analytics chatbot powered by LangChain and OpenAI that can answer complex queries about IPL 2025 cricket data. This chatbot provides detailed analysis of batting, bowling, fielding, partnerships, and team performance.

## 🚀 Features

### 🎯 **Simple Queries**
- Most runs, wickets, fours, and sixes
- Team statistics and rankings
- Match results and venue analysis
- Player performance summaries

### 🚀 **Advanced Batting Analysis**
- Best batters in death overs (16-20) vs pace/spin bowling
- Performance analysis vs specific bowling types
- Strike rates and boundary analysis
- Batting performance in specific overs ranges
- Performance vs left-handed/right-handed batsmen

### 🎾 **Advanced Bowling Analysis**
- Best bowlers in death overs vs RHB/LHB
- Economy rates in specific overs
- Wicket-taking analysis by overs
- Bowling performance vs specific batting hands
- Phase-wise bowling statistics

### 🏃‍♂️ **Fielding & Partnerships**
- Catches, run-outs, and stumpings statistics
- Best batting partnerships
- Team fielding performance
- Individual fielder rankings

### 📊 **Phase-wise Analysis**
- **Powerplay (1-6 overs)**: Batting and bowling statistics
- **Middle overs (7-15)**: Performance analysis
- **Death overs (16-20)**: Critical phase statistics

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cricket-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Groq API (FREE & FAST!)**
   - Get a FREE API key from [Groq Console](https://console.groq.com/)
   - Create a `.env` file in the project root:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

3. **Set up Ollama (Free LLM)**
   ```bash
   # Run the setup script
   python setup_ollama.py
   
   # Or manually:
   # 1. Install Ollama: https://ollama.ai/
   # 2. Start Ollama: ollama serve
   # 3. Pull a model: ollama pull llama2
   ```

## 🤖 Running the Chatbot

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   Open your browser and go to `http://localhost:8501`

## ☁️ Deployment

### Streamlit Cloud
This app can be easily deployed to Streamlit Cloud since it uses the Groq API:

1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and select your forked repository
4. Add your `GROQ_API_KEY` as a secret in the app settings
5. Deploy!

## 💰 Why Groq is Amazing?

- **🆓 FREE Tier**: 14,400 requests/day completely free
- **⚡ Ultra-fast**: Fastest inference in the market
- **💰 Cheap**: 70% cheaper than OpenAI ($0.27-$0.59 per 1M tokens)
- **🤖 Top Models**: Access to Llama 3.1 70B, Mixtral 8x7B, Gemma
- **🔒 No Rate Limits**: In free tier for reasonable usage
- **📈 Scalable**: Easy to upgrade when needed

## 📊 Dataset Overview

The chatbot uses a comprehensive IPL 2025 dataset with the following key information:

- **17,176 ball-by-ball records**
- **10 IPL teams** including:
  - Gujarat Titans
  - Punjab Kings
  - Mumbai Indians
  - Lucknow Super Giants
  - Royal Challengers Bengaluru
  - Rajasthan Royals
  - Delhi Capitals
  - Kolkata Knight Riders
  - Chennai Super Kings
  - Sunrisers Hyderabad

- **282 unique players**
- **Detailed statistics** including:
  - Batting performance (runs, strike rate, boundaries)
  - Bowling performance (wickets, economy, extras)
  - Fielding statistics (catches, run-outs, stumpings)
  - Match context (overs, phases, partnerships)

## 💬 Example Queries

### 🎯 Basic Statistics
```
"Who has scored the most runs in IPL 2025?"
"Who has taken the most wickets?"
"Who has hit the most fours and sixes?"
"Show me the top strike rates"
```

### 🚀 Advanced Batting
```
"Who are the best batters in death overs vs pace bowling?"
"Show me best batters vs right-handed batsmen in overs 16-20"
"Which batters perform best vs spin bowling in middle overs?"
"Top batters in powerplay with highest strike rates"
```

### 🎾 Advanced Bowling
```
"Who are the best bowlers in death overs vs right-handed batsmen?"
"Show me bowlers with best economy rates in overs 16-20"
"Best bowlers vs left-handed batsmen in powerplay"
"Which bowlers take most wickets in middle overs?"
```

### 📊 Team Analysis
```
"What are the team run rates in IPL 2025?"
"Show me powerplay statistics for all teams"
"Which team has the best death overs performance?"
"Team batting and bowling rankings"
```

### 🏃‍♂️ Fielding & Partnerships
```
"Show me the best fielders with most catches and run-outs"
"What are the best batting partnerships in IPL 2025?"
"Fielding statistics by team"
"Partnership analysis by overs"
```

### 📈 Phase-wise Analysis
```
"Show me powerplay statistics for all teams"
"Middle overs batting and bowling performance"
"Death overs analysis - best performers"
"Phase-wise team rankings"
```

## 🛠️ Technical Architecture

### Core Components

1. **LangChain Agent**: Uses structured tools for reliable data analysis
2. **OpenAI GPT-4**: Powers natural language understanding and response generation
3. **Pandas DataFrame**: Handles data manipulation and aggregation
4. **Streamlit**: Provides the web interface

### Key Functions

The chatbot includes 25+ specialized functions for cricket analysis:

#### Basic Statistics
- `get_most_runs()`: Top run scorers
- `get_most_wickets()`: Top wicket takers
- `get_most_fours_and_sixes()`: Boundary statistics

#### Advanced Batting
- `best_batters_death_overs()`: Death overs batting analysis
- `best_batters_vs_batting_hand()`: Performance vs batting hand
- `runs_by_overs_and_style()`: Overs-specific batting analysis

#### Advanced Bowling
- `best_bowlers_death_overs()`: Death overs bowling analysis
- `best_bowlers_vs_batting_hand()`: Bowling vs batting hand
- `bowler_economy()`: Economy rate analysis

#### Fielding & Partnerships
- `get_fielding_stats()`: Comprehensive fielding statistics
- `get_best_partnerships()`: Partnership analysis

#### Team Analysis
- `team_total_runs()`: Team batting statistics
- `team_run_rate()`: Team run rates
- `get_powerplay_stats()`: Powerplay analysis
- `get_middle_overs_stats()`: Middle overs analysis
- `get_death_overs_stats()`: Death overs analysis

## 🎨 User Interface

The chatbot features a modern, intuitive interface with:

- **Chat-based interaction**: Natural conversation flow
- **Example queries sidebar**: Quick access to common questions
- **Streaming responses**: Real-time answer generation
- **Markdown formatting**: Rich text responses with tables
- **Error handling**: Graceful error management
- **Mobile responsive**: Works on all devices

## 🔧 Configuration

### Free LLM Setup
- **Ollama**: Completely free local LLM inference
- **Default Model**: llama2 (can be changed to mistral, codellama, etc.)
- **Temperature**: 0 (for consistent responses)
- **No API Keys Required**: Runs entirely locally

### Data Processing
- **Caching**: 2-hour cache for improved performance
- **Data Cleaning**: Automatic handling of missing values
- **Type Conversion**: Proper data type handling

## 🚀 Usage Tips

1. **Be Specific**: Ask detailed questions for better results
   - ✅ "Best batters in death overs vs pace bowling"
   - ❌ "Who is the best batter?"

2. **Use Cricket Terminology**: The chatbot understands cricket terms
   - "death overs", "powerplay", "strike rate", "economy rate"

3. **Specify Overs**: Mention specific overs for detailed analysis
   - "overs 16-20", "powerplay", "middle overs"

4. **Compare Players/Teams**: Ask for comparisons
   - "Compare batting performance of top 5 run scorers"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **IPL 2025 Data**: Comprehensive ball-by-ball cricket data
- **LangChain**: Framework for building LLM applications
- **Ollama**: Free local LLM inference
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation library

## 📞 Support

For questions or issues:
1. Check the example queries in the sidebar
2. Review the documentation above
3. Open an issue on GitHub

---

**🏏 Ask me anything about IPL 2025 cricket!** 🏏
