# Intelligent Rubric-Based Speech Scoring System
Semantic Similarity • NER • Grammar Heuristics • Sentiment • Vocabulary Analysis

This project is a complete AI-powered rubric evaluation system built for the AI-Nirmaan Case Study.
It automatically evaluates a student’s spoken/self-introduction transcript using:

Semantic similarity (Sentence-BERT)

Named Entity Recognition (spaCy)

Sentiment analysis (HuggingFace Transformers)

Grammar checking (LanguageTool / heuristic fallback)

Vocabulary richness & filler-word analysis

Speech rate computation (text or extracted from audio)

Rule-based and ML-augmented scoring

The system is fully modular, scalable, and optimized for fast local scoring.

# 1. Architecture Diagram
                   +-------------------------+
                   |       User Input        |
                   |  (text / txt / audio)   |
                   +------------+------------+
                                |
                                v
                    +-----------------------+
                    |   Pre-processing      |
                    |  - Text cleaning      |
                    |  - Token extraction   |
                    |  - Audio→duration     |
                    +-----------+-----------+
                                |
                                v
            +------------------------------------------+
            |       Machine Learning Components        |
            |------------------------------------------|
            | 1. Semantic Embeddings (SBERT)           |
            | 2. Named Entity Recognition (spaCy)      |
            | 3. Sentiment Analysis (DistilBERT)       |
            | 4. Grammar Check (LanguageTool)          |
            | 5. Vocabulary Diversity (TTR)            |
            +------------------+-----------------------+
                               |
                               v
            +------------------------------------------+
            |      Rule + ML Hybrid Scoring Engine     |
            |------------------------------------------|
            | - Content & keyword matching              |
            | - Flow/Structure scoring                  |
            | - Speech rate scoring                     |
            | - Grammar & Vocabulary scoring            |
            | - Clarity (filler words)                  |
            | - Engagement (sentiment)                  |
            +------------------+------------------------+
                               |
                               v
               +-------------------------------+
               |       Final Output JSON       |
               | (per-criterion + overall)     |
               +---------------+---------------+
                               |
                          Streamlit UI

# 2. Scoring Formula (Illustrated)

Criterion	Description
Content & Structure	Greeting → Basics → Optional info → Closing
Speech Rate	Based on WPM mapped to rubric
Grammar	LanguageTool or heuristic
Vocabulary	Type-Token Ratio (TTR)
Clarity	Filler word density
Engagement	Positive sentiment probability
Final Score
overall_score =
    Content_Score
  + Speech_Rate_Score
  + Grammar_Score
  + Vocabulary_Score
  + Clarity_Score
  + Engagement_Score

Each component is computed using rule-based + ML hybrid logic.

# 3. Project Structure
NIRMAAN_AI_PROJECT/
│
├── data/
│   └── rubric.xlsx        # Optional rubric file
│
├── app.py                 # Streamlit scoring application
├── requirements.txt
├── README.md

# 4. Installation & Running Instructions
**Step 1 — Clone Repository**<br>
git clone https://github.com/chinthan59/Nirmaan-AI-Case_Study.git <br>
cd NIRMAAN_AI_PROJECT

**Step 2 — Create Virtual Environment**
python -m venv venv


Activate:<br>
1. Windows:
venv\Scripts\activate


2. macOS/Linux:
source venv/bin/activate

**Step 3 — Install Dependencies**
pip install -r requirements.txt

**Step 4 — Run Application**
streamlit run app.py


The app now runs at:
http://localhost:8501

# 5. Demo Video
Demo Video: https://your-demo-video-link

# 6. Tech Stack

Python 3.9+

Streamlit

SentenceTransformers

spaCy

HuggingFace Transformers

LanguageTool

scikit-learn

NumPy / Pandas

Pydub (optional for audio)
