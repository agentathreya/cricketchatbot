# 🏏 IPL Cricket Analytics Chatbot - Powered by Groq API ⚡

> **Ultra-fast cricket insights with AI - Now 70% cheaper and 10x faster!**

A comprehensive cricket analytics chatbot powered by **Groq API** and LangChain that provides lightning-fast analysis of IPL 2025 cricket data with advanced batting, bowling, fielding, and team performance insights.

## 🔥 **Why Groq API?**

| Feature | Groq | OpenAI GPT-4 | DeepSeek | Claude |
|---------|------|--------------|----------|---------|
| **Speed** | ⚡ **10x faster** | 🐌 Slow | 🐌 Medium | 🐌 Medium |
| **Cost** | 💰 **$0.27/1M tokens** | $10-30/1M | $0.60-2.00/1M | $15-75/1M |
| **Free Tier** | 🆓 **14,400 req/day** | $5 credit | Limited | Very limited |
| **Quality** | 🤖 **Llama 3.1 70B** | GPT-4 | Custom | Claude-3 |

## 🚀 **Features**

### 🎯 **Basic Analytics**
- Top run scorers, wicket takers, boundary hitters
- Team rankings and statistics
- Match results and venue analysis
- Player performance summaries

### 📊 **Advanced Analytics**
- **Death Overs Analysis**: Performance in overs 16-20
- **Powerplay Stats**: Overs 1-6 analysis
- **Middle Overs**: Overs 7-15 performance
- **Phase-wise Comparisons**: Detailed breakdowns

### 🏏 **Specialized Insights**
- Best batters vs pace/spin bowling
- Economy rates by bowling type
- Fielding statistics (catches, run-outs, stumpings)
- Partnership analysis
- Strike rate and boundary analysis

## ⚡ **Quick Start**

### **Option 1: Automated Setup (Recommended)**
```bash
git clone https://github.com/agentathreya/cricketchatbot.git
cd cricketchatbot
python3 setup.py  # Installs everything automatically
```

### **Option 2: Manual Setup**
```bash
# 1. Clone repository
git clone https://github.com/agentathreya/cricketchatbot.git
cd cricketchatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get FREE Groq API key
# Visit: https://console.groq.com/
# Sign up and copy your API key

# 4. Create environment file
echo "GROQ_API_KEY=your_actual_api_key_here" > .env

# 5. Run the app
streamlit run app.py
```

## 🌐 **Deployment Options**

### **🔥 Streamlit Cloud (Easiest)**
1. Fork this repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repo
4. Add `GROQ_API_KEY` in secrets
5. Deploy! ✨

### **🐳 Docker**
```bash
docker build -t cricket-chatbot .
docker run -e GROQ_API_KEY=your_key -p 8501:8501 cricket-chatbot
```

### **🚀 Other Platforms**
- **Render**: Auto-deploys from GitHub
- **Railway**: One-click deployment
- **Heroku**: Uses included Procfile
- **Google Cloud Run**: Serverless deployment

## 📊 **Dataset**

- **17,176 ball-by-ball records** from IPL 2025
- **10 IPL teams** with complete rosters
- **282 unique players** with detailed stats
- **Comprehensive metrics**: Batting, bowling, fielding, partnerships

## 💬 **Example Queries**

```
"Who are the top run scorers in IPL 2025?"
"Show me death overs bowling statistics"
"Which team has the best powerplay performance?"
"Best batting partnerships in the tournament"
"Economy rates for spinners vs pace bowlers"
"Fielding statistics by team"
```

## 🛠️ **Technical Stack**

- **Frontend**: Streamlit with modern chat interface
- **AI Engine**: Groq API with Llama 3.1 70B
- **Framework**: LangChain for structured analysis
- **Data Processing**: Pandas for cricket analytics
- **Deployment**: Multi-platform support

## 📈 **Performance Optimization**

### **Model Selection**
```python
# For speed (ultra-fast responses)
model_name="llama3-8b-8192"

# For quality (current default)
model_name="llama3-70b-8192" 

# For complex reasoning
model_name="mixtral-8x7b-32768"
```

### **Cost Optimization**
- FREE tier: 14,400 requests/day
- Paid tier: Only $0.27-$0.59 per 1M tokens
- Built-in caching for repeated queries
- Optimized prompts for efficiency

## 🔧 **Configuration**

### **Environment Variables**
```bash
GROQ_API_KEY=your_groq_api_key  # Required
```

### **Model Configuration**
```python
# In app.py - customize as needed
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama3-70b-8192",  # Change model here
    temperature=0,                 # Deterministic responses
    max_retries=3,
    timeout=60
)
```

## 🐛 **Troubleshooting**

### **Common Issues**

**1. "GROQ_API_KEY not found"**
```bash
# Solution:
echo "GROQ_API_KEY=your_key_here" > .env
```

**2. "Module not found: langchain_groq"**
```bash
# Solution:
pip install -r requirements.txt
```

**3. "Cricket data not found"**
```bash
# Ensure cricket_data.csv is in project root
ls -la cricket_data.csv
```

**4. Slow responses**
```bash
# Switch to faster model in app.py:
model_name="llama3-8b-8192"
```

## 📊 **Usage Statistics**

Perfect for:
- **Cricket Analytics Teams**: Professional insights
- **Fantasy Sports**: Player performance analysis  
- **Sports Journalists**: Quick statistics and trends
- **Cricket Enthusiasts**: Deep-dive analysis
- **Developers**: Learning LangChain + Groq integration

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test locally
4. Submit a pull request

## 📝 **License**

MIT License - see LICENSE file for details

## 🙏 **Acknowledgments**

- **Groq**: Ultra-fast AI inference platform
- **LangChain**: AI application framework
- **Streamlit**: Web app framework
- **IPL**: Cricket data source

## 🆘 **Support**

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Email**: [Your contact if available]

## 🔗 **Links**

- **Live Demo**: [Your deployed app URL]
- **Groq Console**: https://console.groq.com/
- **Documentation**: This README
- **Source Code**: https://github.com/agentathreya/cricketchatbot

---

<div align="center">

### 🏏 **Ready for lightning-fast cricket analytics?** ⚡

**Get your FREE Groq API key and deploy in under 5 minutes!**

[🚀 **Deploy Now**](https://streamlit.io/cloud) | [📊 **View Demo**](your-demo-url) | [⭐ **Star Repo**](https://github.com/agentathreya/cricketchatbot)

</div>
