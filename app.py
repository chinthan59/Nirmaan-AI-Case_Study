import streamlit as st
import pandas as pd, numpy as np, re, json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from transformers import pipeline
from io import BytesIO

try:
    import language_tool_python
    LT_AVAILABLE = True
except:
    LT_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except:
    PYDUB_AVAILABLE = False

RUBRIC_PATH_DEFAULT = "data/rubric.xlsx"
EMB_MODEL = "all-MiniLM-L6-v2"
NER_MODEL = "en_core_web_sm"
SENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

@st.cache_resource
def load_models():
    emb = SentenceTransformer(EMB_MODEL)
    try:
        nlp = spacy.load(NER_MODEL)
    except:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", NER_MODEL])
        nlp = spacy.load(NER_MODEL)
    try:
        sentiment = pipeline("sentiment-analysis", model=SENT_MODEL)
    except:
        sentiment = None
    return emb, nlp, sentiment

def read_rubric(path_or_file):
    xls = pd.ExcelFile(path_or_file)
    sheet = None
    for s in xls.sheet_names:
        if s.strip().lower() in ("rubrics","rubric","sheet1"):
            sheet = s
            break
    if sheet is None:
        sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)
    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
    return df

def preprocess(t):
    return " ".join(re.findall(r"\w+['-]?\w*|\w+", str(t).lower()))

def embed(model, texts):
    if not texts:
        return []
    return model.encode(list(texts), convert_to_numpy=True)

def sim(a,b):
    if a is None or b is None:
        return 0.0
    s = cosine_similarity([a],[b])[0][0]
    return float((s+1)/2)

def extract_entities(nlp, text):
    doc = nlp(text)
    out = {}
    for e in doc.ents:
        out.setdefault(e.label_, []).append(e.text)
    return out

def detect_age(text):
    m = re.search(r"\b(\d{1,2})\s*(years old|yrs old|years|yrs)?\b", text.lower())
    return int(m.group(1)) if m else None

def detect_class(text):
    m = re.search(r"\b(class|grade|standard)\s*(\d{1,2})\b", text.lower())
    if m: return m.group(2)
    m2 = re.search(r"\b(\d{1,2})(st|nd|rd|th)\s*(grade|class|standard)?\b", text.lower())
    if m2: return m2.group(1)
    return None

def compute_ttr(text):
    toks = re.findall(r"\w+['-]?\w*|\w+", text.lower())
    return len(set(toks))/len(toks) if toks else 0.0

def grammar_heuristic(text):
    sents = re.split(r'[.!?]+', text)
    sents = [s.strip() for s in sents if s.strip()]
    words = re.findall(r"\w+", text)
    avg = sum(len(re.findall(r"\w+", s)) for s in sents)/len(sents) if sents else 0
    punct = min(1.0, (text.count(",")+text.count("."))/(max(1,len(sents))*2))
    score = max(0.0, min(1.0, 1 - (0.02*max(0, avg-20)) + 0.1*punct))
    return score

def score(transcript, rubric_df, models):
    emb, nlp, sentmodel = models
    text = transcript.strip()
    words = len(re.findall(r"\w+", text))
    txt_emb = emb.encode([text])[0]

    sal = 0
    tl = text.lower()
    if any(p in tl for p in ["i'm excited","feeling great","excited to introduce"]): sal = 5
    elif any(p in tl for p in ["good morning","good afternoon","good evening","hello everyone","hello all"]): sal = 4
    elif any(p in tl for p in ["hi ","hello ","hey "]): sal = 2

    must = ["name","age","school","class","family","hobby","hobbies","interest","goal","goals"]
    good = ["about family","origin","from","ambition","dream","fun fact","interesting","strength","achievement"]

    found_must = 0
    for kw in must:
        if kw in tl or sim(txt_emb, emb.encode(kw)) > 0.55:
            found_must += 1
    found_good = 0
    for kw in good:
        if kw in tl or sim(txt_emb, emb.encode(kw)) > 0.55:
            found_good += 1

    kw_score = (found_must/len(must))*20 + (found_good/len(good))*10

    def idx(patterns):
        x = [tl.find(p) for p in patterns if tl.find(p)>=0]
        return min(x) if x else None

    sal_i = idx(["hello","hi","good morning","good afternoon","good evening"])
    basic_i = idx(["my name","i am","i'm","age","school","class","from"])
    opt_i = idx(["hobby","hobbies","fun fact","interest","enjoy","play","strength","achievement"])
    close_i = idx(["thank you","thanks"])

    flow = 0
    if sal_i and basic_i and close_i and sal_i < basic_i < close_i: flow = 5
    elif sal_i and basic_i: flow = 3
    elif basic_i: flow = 2

    duration = rubric_df.attrs.get("duration_seconds", 0.0)
    if duration > 0:
        wpm = (words/duration)*60
        if wpm>161: sr = 2
        elif 141<=wpm<=160: sr = 6
        elif 111<=wpm<=140: sr = 10
        elif 81<=wpm<=110: sr = 6
        else: sr = 2
    else:
        wpm = None
        sr = 0

    if LT_AVAILABLE:
        tool = language_tool_python.LanguageTool('en-US')
        matches = tool.check(text)
        errors = len(matches)
        ratio = 1 - min((errors/words*100)/10, 1) if words>0 else 0
    else:
        ratio = grammar_heuristic(text)

    if ratio>=0.9: gram = 10
    elif ratio>=0.7: gram = 8
    elif ratio>=0.5: gram = 6
    elif ratio>=0.3: gram = 4
    else: gram = 2

    ttr = compute_ttr(text)
    if ttr>=0.9: vocab=10
    elif ttr>=0.7: vocab=8
    elif ttr>=0.5: vocab=6
    elif ttr>=0.3: vocab=4
    else: vocab=2

    fillers = ["um","uh","like","you know","so","actually","basically","right","i mean","well","kinda","sort of","okay","hmm","ah"]
    fc = sum(len(re.findall(r"\b"+re.escape(f)+r"\b", tl)) for f in fillers)
    fr = (fc/words*100) if words else 0
    if fr<=3: cl=15
    elif fr<=6: cl=12
    elif fr<=9: cl=9
    elif fr<=12: cl=6
    else: cl=3

    if sentmodel:
        out = sentmodel(text[:512])
        pos = out[0]['score'] if out[0]['label']=="POSITIVE" else 1-out[0]['score']
    else:
        pos = 0.5
    if pos>=0.9: eng=15
    elif pos>=0.7: eng=12
    elif pos>=0.5: eng=9
    elif pos>=0.3: eng=6
    else: eng=3

    content = sal + kw_score + flow
    overall = content + sr + gram + vocab + cl + eng

    return {
        "overall_score": round(overall,2),
        "words": words,
        "per_criterion": [
            {"criterion":"Content & Structure","score":round(content,2)},
            {"criterion":"Speech Rate","score":sr,"meta":{"wpm":wpm}},
            {"criterion":"Grammar","score":gram},
            {"criterion":"Vocabulary","score":vocab},
            {"criterion":"Clarity","score":cl},
            {"criterion":"Engagement","score":eng},
        ],
        "meta":{
            "must_found":found_must,
            "good_found":found_good,
            "ttr":round(ttr,3),
            "filler_count":fc,
            "pos_prob":round(pos,3),
            "wpm":wpm
        }
    }

st.set_page_config(page_title="Nirmaan AI — Optimized Scorer", layout="wide")
st.title("Nirmaan AI — Optimized Rubric Scorer (Semantic + NER)")

models = load_models()

col1, col2 = st.columns([3,1])
with col1:
    transcript = st.text_area("Enter transcript text", height=300)
    uploaded_txt = st.file_uploader("Upload transcript (.txt)", type=["txt"])
    uploaded_audio = st.file_uploader("Upload audio (.mp3/.wav)", type=["mp3","wav"])
    if uploaded_txt and not transcript:
        transcript = uploaded_txt.getvalue().decode("utf-8", errors="ignore")

with col2:
    rubric_upload = st.file_uploader("Upload rubric (.xlsx)", type=["xlsx"])
    duration_input = st.number_input("Duration (seconds)", min_value=0.0, value=0.0)
    if uploaded_audio and PYDUB_AVAILABLE:
        audio = AudioSegment.from_file(BytesIO(uploaded_audio.read()))
        duration = len(audio)/1000
    else:
        duration = duration_input

if st.button("Score"):
    if not transcript.strip():
        st.error("Enter a transcript.")
    else:
        rubric_file = rubric_upload if rubric_upload else RUBRIC_PATH_DEFAULT
        df = read_rubric(rubric_file)
        df.attrs["duration_seconds"] = duration
        result = score(transcript, df, models)
        st.success(f"Overall Score: {result['overall_score']}")
        st.write("Words:", result["words"])
        st.dataframe(pd.DataFrame(result["per_criterion"]))
        st.json(result["meta"])
        st.download_button("Download JSON", data=json.dumps(result,indent=2), file_name="score.json")
