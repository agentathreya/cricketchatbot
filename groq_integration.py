"""
Groq Integration for Cricket Chatbot
====================================

Groq offers fast, cheap inference with excellent models.
Free tier: 14,400 requests/day
Paid: $0.27-$0.59 per 1M tokens (much cheaper than OpenAI)
"""

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def get_groq_llm(model="llama3-70b-8192", temperature=0):
    """
    Initialize Groq LLM
    
    Available models:
    - llama3-70b-8192 (Best performance)
    - llama3-8b-8192 (Faster, good quality)
    - mixtral-8x7b-32768 (Good for complex reasoning)
    - gemma-7b-it (Lightweight)
    """
    return ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name=model,
        temperature=temperature,
        max_retries=3,
        timeout=30
    )

# Usage in app.py:
# Replace line 89-94 with:
# llm = get_groq_llm("llama3-70b-8192")
