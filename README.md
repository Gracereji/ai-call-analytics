I Call Analytics Dashboard

Project Overview

AI Call Analytics Dashboard is a Streamlit-based web application that analyzes customer support call transcripts. It uses AI to provide:

Sentiment Analysis – Detect positive, negative, or neutral tones
Conversation Scoring – Evaluate customer support quality
Summary & Feedback – Highlights key issues, suggestions, and outcomes
Metrics Visualization – Interactive charts for precision, recall, F1-score, and more

This project demonstrates a complete end-to-end AI solution for customer service analytics.

Features
Upload call transcripts or audio files
View sentiment, call outcome, and agent vs. customer analysis
Get actionable AI feedback on call handling
Interactive charts for evaluating metrics and performance
Tech Stack
Frontend: Streamlit
Backend / AI: Python, LangChain, OpenAI, Whisper, Sentence-Transformers
Database / Vector Store: Pinecone
Deployment: Streamlit Community Cloud
Installation

Clone the repository:

git clone https://github.com/Gracereji/ai-call-analytics.git
cd ai-call-analytics

Create and activate a virtual environment:

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Setup Secrets

Create .streamlit/secrets.toml and add your API keys:

GROQ_API_KEY = "your_groq_key"
PINECONE_API_KEY = "your_pinecone_key"
OPENAI_API_KEY = "your_openai_key"

Note: Keep this file private and do not push secrets to GitHub.

Usage

Run the app locally:

streamlit run app.py

Then open http://localhost:8501 in your browser.

Live Demo

Access the live deployed app here:
https://ai-call-analytics-xhdbozctzbym3j4nfrevba.streamlit.app/

Screenshots

(You can add screenshots or GIF demos here to make your README more attractive)

License

This project is open source and free to use for educational and non-commercial purposes.
