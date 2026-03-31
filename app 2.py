"""
app.py — MedAI Clinical Triage System
Multi-symptom selection + rich AI assistant (duration, type, onset, severity).
Run: streamlit run app.py
"""

from pipeline import SymptomPipeline
import streamlit as st

st.set_page_config(
    page_title="MedAI Clinical Triage",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap');

*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;-webkit-font-smoothing:antialiased;}

.stApp{
  background:#070d1a !important;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -20%,rgba(0,82,204,.10),transparent),
    radial-gradient(ellipse 50% 40% at 90% 80%, rgba(0,163,255,.05),transparent);
}
.main .block-container{background:transparent !important;padding:1.5rem 2rem 3rem;max-width:1400px;}
section[data-testid="stSidebar"]{display:none;}

/* NAV */
.topnav{display:flex;align-items:center;justify-content:space-between;
  background:rgba(255,255,255,.03);border:1px solid rgba(0,120,255,.12);
  border-radius:16px;padding:.9rem 1.8rem;margin-bottom:1.6rem;
  box-shadow:0 4px 24px rgba(0,0,0,.4);position:relative;}
.topnav::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#0052cc,#00aaff,#00d4aa);border-radius:16px 16px 0 0;}
.nav-logo{width:38px;height:38px;border-radius:10px;
  background:linear-gradient(135deg,#0052cc,#00aaff);
  display:flex;align-items:center;justify-content:center;font-size:1.2rem;
  box-shadow:0 4px 14px rgba(0,120,255,.4);}
.nav-title{font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;color:#e8f0fe !important;}
.nav-sub{font-size:.68rem;color:#2a5a8a !important;letter-spacing:.1em;text-transform:uppercase;}
.nav-right{display:flex;gap:.6rem;align-items:center;}
.nbadge{padding:.28rem .85rem;border-radius:20px;font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;}
.nb-green{background:rgba(0,208,132,.1);border:1px solid rgba(0,208,132,.3);color:#00d084 !important;}
.nb-blue {background:rgba(0,120,255,.1);border:1px solid rgba(0,120,255,.3);color:#4da6ff !important;}
.nb-grey {background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);color:#3a6a8a !important;}

/* LOGIN */
.login-wrap{max-width:420px;margin:3rem auto;
  background:rgba(255,255,255,.03);border:1px solid rgba(0,120,255,.15);
  border-radius:20px;padding:2.5rem;box-shadow:0 8px 40px rgba(0,0,0,.5);}
.login-title{font-family:'Playfair Display',serif;font-size:1.6rem;color:#e8f0fe !important;margin-bottom:.3rem;}
.login-sub{font-size:.82rem;color:#2a5a8a !important;margin-bottom:1.6rem;}

/* CARDS */
.ccard{background:rgba(255,255,255,.025);border:1px solid rgba(0,100,220,.12);
  border-radius:16px;padding:1.3rem 1.5rem;margin-bottom:1rem;
  box-shadow:0 2px 14px rgba(0,0,0,.3);}
.ccard-header{display:flex;align-items:center;gap:.6rem;margin-bottom:1rem;
  padding-bottom:.7rem;border-bottom:1px solid rgba(0,100,220,.1);}
.ccard-icon{width:30px;height:30px;border-radius:8px;
  display:flex;align-items:center;justify-content:center;font-size:.9rem;}
.ci-blue  {background:rgba(0,82,204,.15);  border:1px solid rgba(0,120,255,.2);}
.ci-green {background:rgba(0,180,100,.12); border:1px solid rgba(0,200,120,.2);}
.ci-amber {background:rgba(220,160,0,.12); border:1px solid rgba(245,180,0,.2);}
.ci-purple{background:rgba(120,60,200,.12);border:1px solid rgba(140,80,220,.2);}
.ci-red   {background:rgba(200,40,40,.12); border:1px solid rgba(220,60,60,.2);}
.ci-cyan  {background:rgba(0,180,200,.12); border:1px solid rgba(0,200,220,.2);}
.ccard-title{font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#3a7ab8 !important;}

/* SYMPTOM CHECKBOX GRID */
.sym-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:.4rem;margin-bottom:.8rem;}
.sym-box{display:flex;align-items:center;gap:.5rem;
  background:rgba(0,40,100,.15);border:1px solid rgba(0,80,180,.15);
  border-radius:8px;padding:.45rem .7rem;cursor:pointer;transition:all .2s;}
.sym-box:hover{border-color:rgba(0,120,255,.4);background:rgba(0,60,150,.2);}
.sym-box.checked{background:rgba(0,82,204,.2);border-color:rgba(0,120,255,.5);
  box-shadow:0 0 0 1px rgba(0,120,255,.2);}
.sym-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.sym-dot-on {background:#00aaff;box-shadow:0 0 6px #00aaff;}
.sym-dot-off{background:rgba(255,255,255,.15);}
.sym-label{font-size:.78rem;color:#8ab8d8 !important;font-weight:500;line-height:1.2;}
.sym-box.checked .sym-label{color:#c0daf8 !important;font-weight:600;}

/* DURATION / TYPE PILLS */
.pill-row{display:flex;flex-wrap:wrap;gap:.4rem;margin:.5rem 0;}
.pill{padding:.3rem .8rem;border-radius:20px;font-size:.75rem;font-weight:600;
  cursor:pointer;transition:all .2s;border:1.5px solid rgba(0,80,180,.2);
  background:rgba(0,30,80,.3);color:#4a7ab8 !important;}
.pill:hover{border-color:rgba(0,150,255,.4);color:#7ab8e8 !important;}
.pill.sel{background:linear-gradient(135deg,#003d99,#0060cc);
  border-color:rgba(0,150,255,.5);color:#a8d0ff !important;
  box-shadow:0 2px 10px rgba(0,100,255,.25);}

/* AI CHAT */
.chat-outer{border:1px solid rgba(0,100,220,.12);border-radius:14px;
  background:rgba(0,10,30,.3);overflow:hidden;}
.chat-log{max-height:340px;overflow-y:auto;padding:1rem;
  scrollbar-width:thin;scrollbar-color:rgba(0,100,220,.2) transparent;}
.chat-msg{display:flex;gap:.7rem;margin-bottom:1rem;align-items:flex-start;}
.chat-avatar{width:32px;height:32px;border-radius:10px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:.9rem;}
.av-bot {background:linear-gradient(135deg,#0052cc,#00aaff);}
.av-user{background:linear-gradient(135deg,#2d2060,#4a30a0);}
.chat-bubble{padding:.7rem 1rem;border-radius:0 12px 12px 12px;
  font-size:.84rem;line-height:1.6;max-width:88%;}
.cb-bot{background:rgba(0,60,160,.12);border:1px solid rgba(0,100,220,.18);color:#8ab8d8 !important;}
.cb-user{background:rgba(60,40,130,.2);border:1px solid rgba(80,60,160,.25);
  color:#a090d0 !important;border-radius:12px 0 12px 12px;margin-left:auto;}
.chat-user-row{flex-direction:row-reverse;}
/* AI structured message parts */
.ai-section{margin:.5rem 0 .3rem;padding:.6rem .8rem;border-radius:8px;}
.ai-sec-diagnosis{background:rgba(0,82,204,.1);border-left:3px solid #0052cc;}
.ai-sec-risk     {background:rgba(220,40,40,.08);border-left:3px solid #ef4444;}
.ai-sec-duration {background:rgba(0,160,120,.08);border-left:3px solid #00c878;}
.ai-sec-advice   {background:rgba(180,120,0,.08); border-left:3px solid #f59e0b;}
.ai-sec-otc      {background:rgba(80,40,180,.08); border-left:3px solid #8b5cf6;}
.ai-sec-label{font-size:.65rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;
  opacity:.7;margin-bottom:.3rem;}
.ai-sec-val{font-size:.84rem;color:#c0daf8 !important;line-height:1.55;}
.ai-meta-row{display:flex;gap:1rem;flex-wrap:wrap;margin:.6rem 0;padding:.6rem .8rem;
  background:rgba(0,40,100,.15);border-radius:8px;border:1px solid rgba(0,80,180,.12);}
.ai-meta{text-align:center;}
.ai-meta-val{font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:500;color:#4da6ff !important;}
.ai-meta-lbl{font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;color:#1a4a7a !important;}
.chat-input-row{display:flex;gap:.5rem;padding:.8rem 1rem;
  border-top:1px solid rgba(0,100,220,.1);background:rgba(0,10,30,.4);}

/* QUICK REPLY CHIPS */
.qr-row{display:flex;flex-wrap:wrap;gap:.4rem;padding:.6rem 1rem;
  border-top:1px solid rgba(0,80,160,.08);}
.qr-chip{padding:.28rem .75rem;border-radius:20px;font-size:.73rem;font-weight:600;
  cursor:pointer;background:rgba(0,50,120,.2);border:1px solid rgba(0,100,220,.2);
  color:#4a8ab8 !important;transition:all .2s;}
.qr-chip:hover{background:rgba(0,80,180,.25);border-color:rgba(0,150,255,.35);color:#7ab8e8 !important;}

/* RISK METER */
.risk-meter{text-align:center;padding:.5rem 0 .8rem;}
.risk-num{font-family:'IBM Plex Mono',monospace;font-size:3rem;font-weight:500;line-height:1;}
.risk-lbl{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:#2a5a8a !important;margin-top:.3rem;}
.risk-bar-wrap{height:10px;background:rgba(255,255,255,.07);border-radius:5px;overflow:hidden;margin:.8rem 0 .3rem;}
.risk-bar-fill{height:100%;border-radius:5px;transition:width .6s ease;}

/* TIMELINE */
.tl-item{display:flex;align-items:flex-start;gap:.7rem;padding:.6rem .8rem;
  margin-bottom:.45rem;background:rgba(0,40,100,.12);
  border:1px solid rgba(0,80,180,.15);border-radius:10px;}
.tl-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;margin-top:4px;}
.tl-body{flex:1;}
.tl-part{font-size:.82rem;font-weight:700;color:#c0daf8 !important;margin-bottom:.2rem;}
.tl-syms{font-size:.75rem;color:#4a7a9b !important;line-height:1.4;}
.tl-meta{font-size:.68rem;color:#1a4a7a !important;margin-top:.25rem;}
.tl-right{text-align:right;flex-shrink:0;}
.tl-sev{font-family:'IBM Plex Mono',monospace;font-size:.82rem;color:#4da6ff !important;font-weight:500;}
.tl-badge{font-size:.65rem;font-weight:700;padding:.15rem .5rem;border-radius:6px;display:block;margin-top:.2rem;}
.tl-low {background:rgba(0,200,120,.12);color:#00d084 !important;}
.tl-med {background:rgba(245,158,11,.12);color:#fbbf24 !important;}
.tl-high{background:rgba(239,68,68,.12); color:#f87171 !important;}

/* RESULT HERO */
.result-hero{border-radius:18px;overflow:hidden;margin-bottom:1.4rem;
  box-shadow:0 12px 40px rgba(0,0,0,.5);}
.rh-high  {background:linear-gradient(135deg,#1a0505,#3d0a0a,#2d0606);border:1px solid rgba(239,68,68,.25);}
.rh-medium{background:linear-gradient(135deg,#140c00,#3d2200,#291500);border:1px solid rgba(245,158,11,.25);}
.rh-low   {background:linear-gradient(135deg,#010f06,#0a2d15,#051a0c);border:1px solid rgba(34,197,94,.25);}
.rh-top{padding:1.8rem 2rem;position:relative;}
.rh-badge{display:inline-flex;align-items:center;gap:.4rem;padding:.28rem .85rem;
  border-radius:20px;font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.7rem;}
.rb-high{background:rgba(239,68,68,.15);border:1.5px solid rgba(239,68,68,.4);color:#f87171 !important;}
.rb-med {background:rgba(245,158,11,.15);border:1.5px solid rgba(245,158,11,.4);color:#fbbf24 !important;}
.rb-low {background:rgba(34,197,94,.15); border:1.5px solid rgba(34,197,94,.4); color:#4ade80 !important;}
.rh-disease{font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:700;
  color:#f0f6ff !important;margin-bottom:.4rem;line-height:1.1;}
.rh-action{font-size:.95rem;color:rgba(240,246,255,.8) !important;font-weight:500;}
.rh-spec  {font-size:.78rem;color:rgba(160,200,240,.55) !important;margin-top:.3rem;}
.conf-box{position:absolute;top:1.6rem;right:1.8rem;background:rgba(0,0,0,.3);
  border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:.8rem 1.2rem;text-align:center;min-width:80px;}
.conf-num{font-family:'IBM Plex Mono',monospace;font-size:1.6rem;font-weight:500;color:#f0f6ff !important;}
.conf-lbl{font-size:.6rem;color:rgba(160,200,240,.4) !important;text-transform:uppercase;letter-spacing:.1em;}
.rh-bottom{padding:.8rem 2rem;background:rgba(0,0,0,.25);
  border-top:1px solid rgba(255,255,255,.05);display:flex;gap:2rem;flex-wrap:wrap;}
.rh-meta{font-size:.73rem;color:rgba(160,200,240,.4) !important;}
.rh-meta span{color:rgba(200,225,255,.75) !important;font-weight:600;}

/* DIFF BARS */
.drow{display:flex;align-items:center;gap:.7rem;margin-bottom:.55rem;}
.dr-rank{font-size:.68rem;color:#1a4a7a !important;min-width:18px;font-weight:700;}
.dr-name{flex:1;font-size:.82rem;color:#6a9abf !important;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.dr-name.top{font-weight:700;color:#c0daf8 !important;}
.dr-track{width:80px;height:5px;background:rgba(0,80,180,.2);border-radius:3px;overflow:hidden;flex-shrink:0;}
.dr-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,#0052cc,#00aaff);}
.dr-fill.top{background:linear-gradient(90deg,#00aaff,#00d4ff);}
.dr-pct{font-family:'IBM Plex Mono',monospace;font-size:.73rem;color:#1a4a7a !important;min-width:36px;text-align:right;}
.dr-pct.top{color:#4da6ff !important;}

/* OTC */
.otc-row{display:flex;gap:.7rem;padding:.5rem 0;border-bottom:1px solid rgba(0,80,180,.08);font-size:.82rem;}
.otc-row:last-child{border-bottom:none;}
.otc-n{font-family:'IBM Plex Mono',monospace;font-size:.68rem;color:#1a4a7a !important;min-width:20px;margin-top:2px;}
.otc-t{color:#6a9abf !important;line-height:1.45;}
.otc-t.w{color:#f87171 !important;}

/* URGENCY */
.urg{border-radius:8px;padding:.6rem 1rem;display:flex;align-items:center;gap:.7rem;margin-bottom:.9rem;font-weight:700;font-size:.83rem;}
.urg-e{background:rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.25); color:#f87171 !important;}
.urg-m{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.25);color:#fbbf24 !important;}
.urg-n{background:rgba(34,197,94,.1); border:1px solid rgba(34,197,94,.25); color:#4ade80 !important;}

/* CHIP */
.chip{display:inline-block;background:rgba(0,60,140,.2);border:1px solid rgba(0,100,220,.2);
  border-radius:6px;padding:.22rem .65rem;margin:.18rem;font-size:.75rem;color:#7ab8e8 !important;font-weight:600;}

/* SECTION DIVIDER */
.sec-div{font-size:.7rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;
  color:#2a5a8a;margin:1.4rem 0 .8rem;display:flex;align-items:center;gap:.6rem;}
.sec-div::before{content:'';display:inline-block;width:20px;height:2px;
  background:linear-gradient(90deg,#0052cc,transparent);border-radius:1px;}

/* DISCLAIMER */
.disc{background:rgba(180,120,0,.06);border:1px solid rgba(180,120,0,.15);
  border-radius:10px;padding:.8rem 1rem;font-size:.76rem;color:#8a6a20 !important;
  margin-top:1rem;display:flex;gap:.6rem;line-height:1.6;}

/* STREAMLIT OVERRIDES */
.stButton>button{
  background:linear-gradient(135deg,#0052cc,#0078ff) !important;
  color:white !important;border:none !important;border-radius:10px !important;
  padding:.6rem 1.6rem !important;font-weight:700 !important;font-size:.85rem !important;
  width:100% !important;letter-spacing:.05em !important;text-transform:uppercase !important;
  box-shadow:0 4px 18px rgba(0,120,255,.3) !important;}
.stButton>button:hover{box-shadow:0 6px 24px rgba(0,120,255,.5) !important;}
div[data-testid="stTextInput"]>div>div{background:rgba(0,20,50,.6) !important;
  border:1px solid rgba(0,120,255,.2) !important;border-radius:10px !important;}
div[data-testid="stTextInput"] input{color:#c0d8f0 !important;}
div[data-baseweb="select"]>div{background:rgba(0,20,50,.6) !important;
  border:1px solid rgba(0,120,255,.2) !important;border-radius:10px !important;}
div[data-baseweb="select"] input{color:#c0d8f0 !important;}
div[data-baseweb="menu"]{background:#060f1e !important;border:1px solid rgba(0,120,255,.18) !important;border-radius:10px !important;}
div[data-baseweb="menu"] li{color:#8ab0d0 !important;font-size:.86rem !important;}
div[data-baseweb="menu"] li:hover{background:rgba(0,120,255,.1) !important;}
.stMultiSelect [data-baseweb="tag"]{background:linear-gradient(135deg,#003d99,#0060cc) !important;border-radius:6px !important;}
.stMultiSelect [data-baseweb="tag"] span{color:#a8d0ff !important;}
div[data-testid="stSlider"] *{color:#4a7a9b !important;}
.stCheckbox label{color:#6a9abf !important;font-size:.84rem !important;}
.stCheckbox [data-testid="stCheckbox"] input:checked+div{background:#0052cc !important;border-color:#0078ff !important;}
p,li,span,label{color:#6a9abf !important;}
h1,h2,h3,h4,h5{color:#e8f0fe !important;}
.stCaption,.stCaption *{color:#1a3a5a !important;font-size:.71rem !important;}
.stMarkdown h4{color:#3a7ab8 !important;font-size:.71rem !important;font-weight:800 !important;
  letter-spacing:.1em !important;text-transform:uppercase !important;}
hr{border-color:rgba(0,100,220,.1) !important;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════
BODY_SYMPTOMS = {
    "Head":     ["headache","frontal headache","dizziness","disturbance of memory",
                 "seizures","slurring words","neck stiffness or tightness","blurred vision"],
    "Eyes":     ["eye redness","pain in eye","diminished vision","double vision",
                 "itchiness of eye","eye burns or stings","lacrimation","swollen eye"],
    "Ears/Nose":["ear pain","ringing in ear","nasal congestion","sneezing",
                 "nosebleed","sore throat","hoarse voice","fluid in ear"],
    "Chest":    ["sharp chest pain","chest tightness","palpitations","shortness of breath",
                 "breathing fast","wheezing","cough","irregular heartbeat","congestion in chest"],
    "Abdomen":  ["upper abdominal pain","lower abdominal pain","sharp abdominal pain",
                 "nausea","vomiting","diarrhea","constipation","stomach bloating","heartburn","jaundice"],
    "Back":     ["back pain","low back pain","neck pain","rib pain",
                 "muscle stiffness or tightness","back weakness","low back weakness"],
    "Arms":     ["arm pain","arm weakness","arm swelling","wrist pain",
                 "hand or finger pain","hand or finger swelling","joint pain","muscle pain"],
    "Legs":     ["leg pain","leg weakness","leg swelling","knee pain",
                 "ankle pain","foot or toe pain","joint stiffness or tightness","leg cramps or spasms"],
    "Skin":     ["skin rash","itching of skin","abnormal appearing skin","skin lesion",
                 "acne or pimples","pallor","sweating","skin dryness, peeling, scaliness, or roughness"],
    "Urinary":  ["frequent urination","painful urination","blood in urine",
                 "involuntary urination","retention of urine","low urine output","polyuria"],
    "Mental":   ["anxiety and nervousness","depression","insomnia","restlessness",
                 "fatigue","disturbance of memory","fears and phobias","excessive anger","sleepiness"],
    "General":  ["fever","chills","fatigue","recent weight loss","weight gain",
                 "feeling cold","feeling ill","ache all over","swollen lymph nodes","pallor"],
}

DURATIONS   = ["< 1 day", "1–3 days", "4–7 days", "1–2 weeks", "2–4 weeks", "1–3 months", "3+ months"]
ONSET_TYPES = ["Sudden (came on quickly)", "Gradual (built up slowly)", "Recurring (comes and goes)"]
SYM_TYPES   = ["Sharp / stabbing", "Dull / aching", "Burning", "Throbbing / pulsing",
               "Pressure / squeezing", "Tingling / numbness", "Constant", "Intermittent"]
TRIGGERS    = ["Physical activity", "Eating / drinking", "Stress / anxiety",
               "Rest / lying down", "Morning (on waking)", "Night / sleep", "No clear trigger"]

DEMO_USERS  = {"doctor@medai.com":"admin123", "patient@medai.com":"pass123"}

# ═══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════
defaults = {
    "logged_in":False, "username":"",
    "selected_parts":[],      # LIST — multiple body parts at once
    "all_picked":{},          # {part: [syms]}
    "severity":5,
    "duration":"1–3 days",
    "onset":"Gradual (built up slowly)",
    "sym_type":"Dull / aching",
    "trigger":"No clear trigger",
    "timeline":[],
    "risk_score":0,
    "messages":[],
    "result":None,
    "top_n":5,
    "quick_reply":None,
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Initialising clinical AI…")
def load_pipeline():
    return SymptomPipeline()
pipeline = load_pipeline()


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════
def sev_colour(s):
    if s<=3: return "#00d084"
    if s<=6: return "#fbbf24"
    return "#f87171"

def sev_label(s):
    if s<=3: return "Mild","tl-low"
    if s<=6: return "Moderate","tl-med"
    return "Severe","tl-high"

def calc_risk(tl):
    if not tl: return 0
    return min(100, round(sum(t["severity"] for t in tl)/len(tl)*10))

def risk_meta(r):
    if r>=70: return "#f87171","rh-high","rb-high","urg-e","🔴 High Risk"
    if r>=40: return "#fbbf24","rh-medium","rb-med","urg-m","🟡 Medium Risk"
    return "#4ade80","rh-low","rb-low","urg-n","🟢 Low Risk"

def bot(text):
    st.session_state.messages.append({"role":"bot","text":text})

def total_symptoms():
    syms = []
    for s_list in st.session_state.all_picked.values():
        syms.extend(s_list)
    # also from timeline
    for entry in st.session_state.timeline:
        syms.extend(entry.get("symptoms",[]))
    return list(dict.fromkeys(syms))

def build_ai_response(result, entry):
    """Build a rich structured AI message from prediction result + symptom entry."""
    d   = result["predicted_disease"].title()
    c   = result["confidence"]
    r   = result["risk_level"]
    u   = result["urgency"]
    otc = result["otc"][:3]
    r_e = {"High":"🔴","Medium":"🟡","Low":"🟢"}[r]
    syms_str   = ", ".join(entry["symptoms"][:4])
    sev_lbl,_  = sev_label(entry["severity"])
    dur        = entry.get("duration","—")
    onset      = entry.get("onset","—")
    sym_type   = entry.get("sym_type","—")

    otc_html = "".join(f"<div style='font-size:.8rem;color:#8ab8d8;padding:.2rem 0;'>• {o}</div>" for o in otc)

    return f"""
<div>
  <div style='font-size:.8rem;color:#4a7ab8;margin-bottom:.6rem;'>
    Analysis complete for <b style='color:#7ab8e8'>{entry['part']}</b> symptoms
  </div>

  <div class='ai-meta-row'>
    <div class='ai-meta'><div class='ai-meta-val'>{c:.0f}%</div><div class='ai-meta-lbl'>Confidence</div></div>
    <div class='ai-meta'><div class='ai-meta-val'>{entry["severity"]}/10</div><div class='ai-meta-lbl'>Severity</div></div>
    <div class='ai-meta'><div class='ai-meta-val'>{dur}</div><div class='ai-meta-lbl'>Duration</div></div>
    <div class='ai-meta'><div class='ai-meta-val'>{r_e}</div><div class='ai-meta-lbl'>Risk</div></div>
  </div>

  <div class='ai-section ai-sec-diagnosis'>
    <div class='ai-sec-label'>🩺 Most Likely Diagnosis</div>
    <div class='ai-sec-val'><b>{d}</b> — {c:.0f}% confidence</div>
  </div>

  <div class='ai-section ai-sec-risk'>
    <div class='ai-sec-label'>⚠️ Risk & Urgency</div>
    <div class='ai-sec-val'>{r_e} <b>{r} Risk</b> · {u}<br>
    <span style='font-size:.78rem;opacity:.8'>{result["note"][:120]}{"…" if len(result["note"])>120 else ""}</span></div>
  </div>

  <div class='ai-section ai-sec-duration'>
    <div class='ai-sec-label'>📅 Symptom Profile</div>
    <div class='ai-sec-val'>
      <b>Symptoms:</b> {syms_str}<br>
      <b>Duration:</b> {dur} &nbsp;·&nbsp; <b>Onset:</b> {onset}<br>
      <b>Type:</b> {sym_type}
    </div>
  </div>

  <div class='ai-section ai-sec-otc'>
    <div class='ai-sec-label'>💊 Immediate Guidance</div>
    {otc_html}
  </div>

  <div class='ai-section ai-sec-advice'>
    <div class='ai-sec-label'>👨‍⚕️ Recommended Action</div>
    <div class='ai-sec-val'>{result["recommendation"]}<br>
    <span style='font-size:.78rem;opacity:.8'>Specialist: {result["specialist"]}</span></div>
  </div>
</div>"""


# ═══════════════════════════════════════════════════════════════════
#  TOPNAV
# ═══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="topnav">
  <div style="display:flex;align-items:center;gap:1rem;">
    <div class="nav-logo">⚕️</div>
    <div>
      <div class="nav-title">MedAI Clinical Triage</div>
      <div class="nav-sub">AI-Powered Symptom Intelligence</div>
    </div>
  </div>
  <div class="nav-right">
    <span class="nbadge nb-green">● {"ONLINE" if pipeline.has_model else "HEURISTIC"}</span>
    <span class="nbadge nb-blue">607 Conditions</span>
    <span class="nbadge nb-grey">377 Symptoms</span>
    {"<span class='nbadge nb-grey'>👤 "+st.session_state.username+"</span>" if st.session_state.logged_in else ""}
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  LOGIN
# ═══════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    _, lc, _ = st.columns([1,1.4,1])
    with lc:
        st.markdown("""
        <div class="login-wrap">
          <div style="text-align:center;margin-bottom:1.5rem;">
            <div style="font-size:2.5rem;margin-bottom:.5rem;">⚕️</div>
            <div class="login-title">Welcome Back</div>
            <div class="login-sub">Sign in to access the clinical triage system</div>
          </div>
        """, unsafe_allow_html=True)
        email    = st.text_input("Email",    placeholder="doctor@medai.com", key="le")
        password = st.text_input("Password", placeholder="••••••••", type="password", key="lp")
        if st.button("SIGN IN →", key="login_btn"):
            if email in DEMO_USERS and DEMO_USERS[email] == password:
                st.session_state.logged_in = True
                st.session_state.username  = email.split("@")[0]
                bot(f"Welcome Dr. **{email.split('@')[0].title()}**! 👋\n\n"
                    "I'm your MedAI clinical assistant. Here's how to use the system:\n\n"
                    "**Step 1** — Select a body area from the map\n"
                    "**Step 2** — Tick all symptoms that apply (multiple allowed)\n"
                    "**Step 3** — Fill in duration, onset type, and severity\n"
                    "**Step 4** — Press **LOG & ANALYZE** to get instant AI diagnosis\n\n"
                    "You can log symptoms from multiple body areas before running the final analysis.")
                st.rerun()
            else:
                st.error("Invalid credentials. Try: doctor@medai.com / admin123")
        st.markdown("""
          <div style="margin-top:1rem;text-align:center;font-size:.75rem;color:#1a3a5a;">
            Demo: <code style="color:#4da6ff">doctor@medai.com</code> / <code style="color:#4da6ff">admin123</code>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════
#  MAIN — 4 COLUMNS
# ═══════════════════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns([1.05, 1.15, 0.95, 1.1])


# ─────────────────────────────
#  COL 1 · BODY MAP
# ─────────────────────────────
with col1:
    st.markdown("""
    <div class="ccard">
      <div class="ccard-header">
        <div class="ccard-icon ci-blue">🫀</div>
        <div class="ccard-title">Body Map — Select Multiple Areas</div>
      </div>
    """, unsafe_allow_html=True)

    parts = st.session_state.selected_parts   # list of selected parts

    # SVG body diagram — highlight ALL selected parts
    def fill(p): return "#00aaff" if p in parts else "#0d2d4a"
    def strk(p): return "#00d4ff" if p in parts else "#0d3a5c"
    def txt(p):  return "white"   if p in parts else "#1a5a8a"

    st.markdown(f"""
    <svg viewBox="0 0 200 420" style="width:100%;max-width:155px;display:block;margin:0 auto .8rem;">
      <circle cx="100" cy="36" r="26" fill="{fill('Head')}" stroke="{strk('Head')}" stroke-width="2"/>
      <text x="100" y="40" text-anchor="middle" fill="{txt('Head')}" font-size="8" font-weight="700" font-family="Inter">HEAD</text>
      <rect x="91" y="62" width="18" height="16" fill="{fill('Head')}" stroke="{strk('Head')}" stroke-width="1.5"/>
      <rect x="60" y="80" width="80" height="86" rx="8" fill="{fill('Chest')}" stroke="{strk('Chest')}" stroke-width="2"/>
      <text x="100" y="128" text-anchor="middle" fill="{txt('Chest')}" font-size="9" font-weight="700" font-family="Inter">CHEST</text>
      <rect x="63" y="168" width="74" height="78" rx="6" fill="{fill('Abdomen')}" stroke="{strk('Abdomen')}" stroke-width="2"/>
      <text x="100" y="212" text-anchor="middle" fill="{txt('Abdomen')}" font-size="9" font-weight="700" font-family="Inter">ABDOMEN</text>
      <rect x="36" y="82" width="22" height="106" rx="10" fill="{fill('Arms')}" stroke="{strk('Arms')}" stroke-width="2"/>
      <rect x="142" y="82" width="22" height="106" rx="10" fill="{fill('Arms')}" stroke="{strk('Arms')}" stroke-width="2"/>
      <text x="47" y="137" text-anchor="middle" fill="{txt('Arms')}" font-size="7" font-family="Inter">ARM</text>
      <text x="153" y="137" text-anchor="middle" fill="{txt('Arms')}" font-size="7" font-family="Inter">ARM</text>
      <rect x="60" y="248" width="32" height="130" rx="10" fill="{fill('Legs')}" stroke="{strk('Legs')}" stroke-width="2"/>
      <rect x="108" y="248" width="32" height="130" rx="10" fill="{fill('Legs')}" stroke="{strk('Legs')}" stroke-width="2"/>
      <text x="76" y="318" text-anchor="middle" fill="{txt('Legs')}" font-size="8" font-family="Inter">LEG</text>
      <text x="124" y="318" text-anchor="middle" fill="{txt('Legs')}" font-size="8" font-family="Inter">LEG</text>
      <rect x="63" y="78" width="4" height="168" rx="2" fill="{fill('Back')}" stroke="{strk('Back')}" stroke-width="0" opacity=".7"/>
    </svg>
    """, unsafe_allow_html=True)

    # Body part buttons — toggle multi-select
    all_parts = list(BODY_SYMPTOMS.keys())
    for i in range(0, len(all_parts), 3):
        row_parts = all_parts[i:i+3]
        cols = st.columns(len(row_parts))
        for bp, c in zip(row_parts, cols):
            picked_count = len(st.session_state.all_picked.get(bp, []))
            is_sel = bp in st.session_state.selected_parts
            if is_sel:
                label = f"✓ {bp}" + (f" ({picked_count})" if picked_count else "")
            else:
                label = f"+ {bp}" + (f" ({picked_count})" if picked_count else "")
            with c:
                if st.button(label, key=f"bp_{bp}"):
                    if bp in st.session_state.selected_parts:
                        st.session_state.selected_parts.remove(bp)
                    else:
                        st.session_state.selected_parts.append(bp)
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Total symptom count badge
    total = sum(len(v) for v in st.session_state.all_picked.values())
    if total > 0:
        st.markdown(f"""
        <div style="background:rgba(0,82,204,.15);border:1px solid rgba(0,120,255,.25);
             border-radius:10px;padding:.5rem .8rem;text-align:center;margin-top:.3rem;">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;color:#4da6ff;">{total}</span>
          <span style="font-size:.72rem;color:#2a5a8a;display:block;text-transform:uppercase;letter-spacing:.08em;">total symptoms</span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────
#  COL 2 · MULTI SYMPTOM + DETAILS
# ─────────────────────────────
with col2:
    st.markdown("""
    <div class="ccard">
      <div class="ccard-header">
        <div class="ccard-icon ci-amber">⚡</div>
        <div class="ccard-title">Symptom Details</div>
      </div>
    """, unsafe_allow_html=True)

    sel_parts = st.session_state.selected_parts

    if sel_parts:
        # Show selected parts as tabs/badges
        badges = "".join(
            f"<span style='background:rgba(0,82,204,.2);border:1px solid rgba(0,120,255,.35);"
            f"border-radius:6px;padding:.2rem .6rem;margin:.2rem;font-size:.75rem;"
            f"color:#4da6ff;display:inline-block;'>{p}</span>"
            for p in sel_parts
        )
        total_sel = sum(len(st.session_state.all_picked.get(p,[])) for p in sel_parts)
        st.markdown(f"""
        <div style="background:rgba(0,120,255,.06);border:1px solid rgba(0,120,255,.15);
             border-radius:10px;padding:.55rem .9rem;margin-bottom:.9rem;">
          <div style="font-size:.68rem;color:#2a5a8a;text-transform:uppercase;
               letter-spacing:.08em;margin-bottom:.4rem;">Selected body areas</div>
          <div>{badges}</div>
          {"<div style='font-size:.72rem;color:#00aaff;margin-top:.4rem;'>✓ " + str(total_sel) + " symptom(s) selected across " + str(len(sel_parts)) + " area(s)</div>" if total_sel else ""}
        </div>
        """, unsafe_allow_html=True)

        # Show symptoms for ALL selected parts grouped by part
        st.markdown("#### Select Symptoms — All Areas")
        all_updated = {}
        for current_part in sel_parts:
            avail   = BODY_SYMPTOMS.get(current_part, [])
            current = st.session_state.all_picked.get(current_part, [])
            st.markdown(f"""
            <div style="font-size:.7rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;
                 color:#3a7ab8;margin:.7rem 0 .4rem;display:flex;align-items:center;gap:.5rem;">
              <span style="width:14px;height:2px;background:#0052cc;display:inline-block;border-radius:1px;"></span>
              {current_part}
            </div>
            """, unsafe_allow_html=True)
            updated_part = []
            cols_cb = st.columns(2)
            for idx, sym in enumerate(avail):
                is_checked = sym in current
                with cols_cb[idx % 2]:
                    if st.checkbox(sym, value=is_checked, key=f"cb_{current_part}_{idx}"):
                        updated_part.append(sym)
            st.session_state.all_picked[current_part] = updated_part
            all_updated[current_part] = updated_part

        # Combined symptom list for logging
        updated  = [s for lst in all_updated.values() for s in lst]
        n_sel    = len(updated)
        if n_sel > 0:
            st.markdown(f"""
            <div style="font-size:.72rem;color:#00aaff;margin:.3rem 0 .8rem;
                 background:rgba(0,120,255,.06);border-radius:6px;padding:.35rem .7rem;
                 border:1px solid rgba(0,120,255,.15);">
              ✓ {n_sel} total symptom{"s" if n_sel>1 else ""} selected across {len(sel_parts)} area(s)
            </div>
            """, unsafe_allow_html=True)
        # keep current_part as last selected for entry label
        current_part = sel_parts[-1]

        st.markdown("---")

        # ── SEVERITY ──
        st.markdown("#### Severity  (1 = Mild · 10 = Severe)")
        st.markdown("""
        <div style="height:6px;border-radius:3px;margin:.3rem 0;
             background:linear-gradient(90deg,#00d084,#f59e0b,#ef4444);"></div>
        """, unsafe_allow_html=True)
        st.session_state.severity = st.slider(
            "sev", 1, 10, st.session_state.severity, label_visibility="collapsed")
        sc = sev_colour(st.session_state.severity)
        sl, _ = sev_label(st.session_state.severity)
        st.markdown(f"""
        <div style="text-align:center;margin:.2rem 0 .7rem;">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:1.6rem;color:{sc};font-weight:500;">
            {st.session_state.severity}/10
          </span>
          <span style="font-size:.75rem;color:{sc};margin-left:.5rem;font-weight:700;">{sl}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── DURATION ──
        st.markdown("#### Duration")
        st.session_state.duration = st.select_slider(
            "dur", options=DURATIONS,
            value=st.session_state.duration, label_visibility="collapsed")

        # ── ONSET TYPE ──
        st.markdown("#### Onset Type")
        st.session_state.onset = st.radio(
            "onset", ONSET_TYPES,
            index=ONSET_TYPES.index(st.session_state.onset),
            label_visibility="collapsed")

        # ── SYMPTOM CHARACTER ──
        st.markdown("#### Symptom Character / Feel")
        st.session_state.sym_type = st.selectbox(
            "sym_type", SYM_TYPES,
            index=SYM_TYPES.index(st.session_state.sym_type),
            label_visibility="collapsed")

        # ── TRIGGER ──
        st.markdown("#### What Makes It Worse?")
        st.session_state.trigger = st.selectbox(
            "trigger", TRIGGERS,
            index=TRIGGERS.index(st.session_state.trigger),
            label_visibility="collapsed")

        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

        # ── LOG & ANALYZE BUTTON ──
        can_log = n_sel > 0
        if st.button("➕  LOG & ANALYZE", key="log_btn", disabled=not can_log):
            parts_label = " + ".join(sel_parts)
            entry = {
                "part":     parts_label,
                "symptoms": list(updated),
                "severity": st.session_state.severity,
                "duration": st.session_state.duration,
                "onset":    st.session_state.onset,
                "sym_type": st.session_state.sym_type,
                "trigger":  st.session_state.trigger,
            }
            st.session_state.timeline.append(entry)
            st.session_state.risk_score = calc_risk(st.session_state.timeline)

            # Run prediction
            all_syms = total_symptoms()
            st.session_state.result = pipeline.predict(all_syms, top_n=st.session_state.top_n)

            # Rich AI response
            ai_msg = build_ai_response(st.session_state.result, entry)
            bot(ai_msg)

            # Clear this part's selection
            st.session_state.all_picked[current_part] = []
            st.rerun()

    else:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 1rem;color:#1a4a7a;font-size:.85rem;line-height:1.8;">
          ← Tap <b style="color:#4da6ff">one or more</b><br>body areas on the map<br><br>
          <span style="font-size:.75rem;color:#0d2a4a;">
            Select multiple areas at once —<br>all their symptoms will appear here
          </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────
#  COL 3 · RISK + TIMELINE
# ─────────────────────────────
with col3:
    rs = st.session_state.risk_score
    rc, rh_cls, rb_cls, urg_cls, r_label = risk_meta(rs)

    # Risk meter
    st.markdown(f"""
    <div class="ccard">
      <div class="ccard-header">
        <div class="ccard-icon ci-red">📊</div>
        <div class="ccard-title">Risk Score</div>
      </div>
      <div class="risk-meter">
        <div class="risk-num" style="color:{rc}">
          {rs}<span style="font-size:1.4rem;opacity:.5">%</span>
        </div>
        <div class="risk-lbl">{r_label}</div>
        <div class="risk-bar-wrap">
          <div class="risk-bar-fill" style="width:{rs}%;
               background:linear-gradient(90deg,#00d084,
               {'#f59e0b' if rs>40 else '#00d084'},
               {'#ef4444' if rs>70 else '#f59e0b'});"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:.63rem;color:#1a3a5a;">
          <span>Low</span><span>Moderate</span><span>High</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Timeline
    st.markdown("""
    <div class="ccard">
      <div class="ccard-header">
        <div class="ccard-icon ci-purple">📋</div>
        <div class="ccard-title">Symptom Timeline</div>
      </div>
    """, unsafe_allow_html=True)

    if st.session_state.timeline:
        for i, entry in enumerate(st.session_state.timeline):
            col_dot = sev_colour(entry["severity"])
            sl, badge_cls = sev_label(entry["severity"])
            syms_preview = ", ".join(entry["symptoms"][:3])
            if len(entry["symptoms"]) > 3:
                syms_preview += f" +{len(entry['symptoms'])-3} more"
            st.markdown(f"""
            <div class="tl-item">
              <div class="tl-dot" style="background:{col_dot};box-shadow:0 0 6px {col_dot};margin-top:4px;"></div>
              <div class="tl-body">
                <div class="tl-part">{entry['part']}</div>
                <div class="tl-syms">{syms_preview}</div>
                <div class="tl-meta">
                  ⏱ {entry.get('duration','—')} &nbsp;·&nbsp;
                  ↗ {entry.get('onset','—').split('(')[0].strip()} &nbsp;·&nbsp;
                  ✦ {entry.get('sym_type','—').split('/')[0].strip()}
                </div>
              </div>
              <div class="tl-right">
                <div class="tl-sev">{entry['severity']}/10</div>
                <span class="tl-badge {badge_cls}">{sl}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Top N
        st.session_state.top_n = st.slider("Top predictions", 3, 10, st.session_state.top_n, key="tn2")

        c_a, c_b = st.columns(2)
        with c_a:
            if st.button("🔍 RE-ANALYZE", key="reanalyze"):
                all_syms = total_symptoms()
                st.session_state.result = pipeline.predict(all_syms, top_n=st.session_state.top_n)
                bot(build_ai_response(st.session_state.result,
                    st.session_state.timeline[-1]))
                st.rerun()
        with c_b:
            if st.button("🗑 CLEAR ALL", key="clear_all"):
                for k in ["timeline","risk_score","result","all_picked","messages","selected_parts"]:
                    st.session_state[k] = {} if k=="all_picked" else ([] if k in ["timeline","messages","selected_parts"] else 0 if k=="risk_score" else None)
                st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center;padding:2rem .5rem;color:#1a4a7a;font-size:.82rem;line-height:1.8;">
          No symptoms logged yet.<br>
          <span style="font-size:.75rem;color:#0d2a4a;">
            Select a body area, tick<br>symptoms, and press LOG.
          </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────
#  COL 4 · AI ASSISTANT
# ─────────────────────────────
with col4:
    st.markdown("""
    <div class="ccard-header" style="margin-bottom:.8rem;">
      <div class="ccard-icon ci-cyan">🤖</div>
      <div class="ccard-title">AI Clinical Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.messages:
        bot("Hello! I'm your MedAI clinical assistant. Select a body area and tick your symptoms to begin. I'll give you a full structured analysis including diagnosis, risk, duration profile, and treatment guidance.")

    # Chat log
    chat_html = '<div class="chat-outer"><div class="chat-log">'
    for msg in st.session_state.messages[-8:]:
        if msg["role"] == "bot":
            chat_html += f"""
            <div class="chat-msg">
              <div class="chat-avatar av-bot">🤖</div>
              <div class="chat-bubble cb-bot">{msg["text"]}</div>
            </div>"""
        else:
            chat_html += f"""
            <div class="chat-msg chat-user-row">
              <div class="chat-avatar av-user">👤</div>
              <div class="chat-bubble cb-user">{msg["text"]}</div>
            </div>"""
    chat_html += "</div>"

    # Quick reply chips
    qr_options = ["What is my risk?", "Show diagnosis", "What symptoms did I log?",
                  "What should I do?", "How to use this?", "Clear chat"]
    qr_html = '<div class="qr-row">'
    for qr in qr_options:
        qr_html += f"<span class='qr-chip' title='{qr}'>{qr}</span>"
    qr_html += "</div></div>"
    st.markdown(chat_html + qr_html, unsafe_allow_html=True)

    # Quick reply buttons (functional)
    qr_cols = st.columns(3)
    qr_map  = {
        "Risk score":   "What is my risk?",
        "Diagnosis":    "Show diagnosis",
        "My symptoms":  "What symptoms did I log?",
        "What to do":   "What should I do?",
        "How to use":   "How to use this?",
        "Clear chat":   "Clear chat",
    }
    for idx, (label, question) in enumerate(qr_map.items()):
        with qr_cols[idx % 3]:
            if st.button(label, key=f"qr_{idx}"):
                st.session_state.messages.append({"role":"user","text":question})
                # Generate answer
                u = question.lower()
                if "risk" in u:
                    rs2 = st.session_state.risk_score
                    _, _, _, _, rl = risk_meta(rs2)
                    bot(f"Your current risk score is <b style='color:#4da6ff'>{rs2}%</b> — <b>{rl}</b>.<br>"
                        f"This is based on {len(st.session_state.timeline)} logged symptom group(s) "
                        f"with average severity {rs2//10}/10.")
                elif "diagnosis" in u or "show" in u:
                    if st.session_state.result:
                        r2 = st.session_state.result
                        top3 = "".join(f"<div style='font-size:.8rem;color:#8ab8d8;padding:.15rem 0;'>#{i+1} {d['disease'].title()} — {d['probability_pct']:.1f}%</div>"
                                       for i,d in enumerate(r2["top_diseases"][:3]))
                        bot(f"<b>Top prediction:</b> <span style='color:#c0daf8'>{r2['predicted_disease'].title()}</span> "
                            f"({r2['confidence']:.0f}% confidence)<br><br>"
                            f"<b>Differential (top 3):</b><br>{top3}<br>"
                            f"<b>Urgency:</b> {r2['urgency']} · <b>Risk:</b> {r2['risk_level']}")
                    else:
                        bot("No analysis run yet. Log your symptoms and press **LOG & ANALYZE** to get a diagnosis.")
                elif "symptom" in u:
                    all_s = total_symptoms()
                    if all_s:
                        parts_used = list({e["part"] for e in st.session_state.timeline})
                        bot(f"You have logged <b>{len(all_s)} symptoms</b> across <b>{len(parts_used)} body area(s)</b>: "
                            f"<i>{', '.join(parts_used)}</i>.<br><br>"
                            f"Symptoms: <span style='color:#7ab8e8'>{', '.join(all_s[:8])}"
                            f"{'…' if len(all_s)>8 else ''}</span>")
                    else:
                        bot("No symptoms logged yet. Select a body area on the map and tick your symptoms.")
                elif "do" in u or "action" in u:
                    if st.session_state.result:
                        r2 = st.session_state.result
                        bot(f"<b>Recommended action:</b><br>{r2['recommendation']}<br><br>"
                            f"<b>Specialist:</b> {r2['specialist']}<br><br>"
                            f"<b>Clinical note:</b><br><span style='font-size:.8rem;color:#6a9abf'>{r2['note']}</span>")
                    else:
                        bot("Please log your symptoms first and press **LOG & ANALYZE** to get personalised action advice.")
                elif "how" in u or "use" in u:
                    bot("<b>How to use MedAI:</b><br><br>"
                        "1️⃣ <b>Select a body area</b> from the map on the left<br>"
                        "2️⃣ <b>Tick all symptoms</b> that apply (multiple allowed)<br>"
                        "3️⃣ <b>Set severity</b> (1–10), duration, onset type & symptom character<br>"
                        "4️⃣ Press <b>LOG & ANALYZE</b> — I'll give a full structured report<br>"
                        "5️⃣ You can log symptoms from <b>multiple body areas</b> for a complete picture<br>"
                        "6️⃣ Use <b>RE-ANALYZE</b> after adding more symptoms<br><br>"
                        "Ask me anything using the quick reply buttons or the text box below.")
                elif "clear" in u:
                    st.session_state.messages = []
                    bot("Chat cleared. How can I help you?")
                st.rerun()

    # Free text input
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    user_input = st.text_input("Ask anything…", placeholder="e.g. What does this diagnosis mean?",
                                label_visibility="collapsed", key="free_chat")
    if st.button("SEND →", key="send_free"):
        if user_input.strip():
            st.session_state.messages.append({"role":"user","text":user_input})
            u = user_input.lower()
            if any(w in u for w in ["risk","danger","serious"]):
                rs2 = st.session_state.risk_score
                _, _, _, _, rl = risk_meta(rs2)
                bot(f"Your risk score is <b>{rs2}%</b> ({rl}). "
                    + ("Please seek emergency care immediately." if rs2>=70
                       else "A doctor visit is recommended soon." if rs2>=40
                       else "You can manage this with home care, but monitor symptoms."))
            elif any(w in u for w in ["diagnosis","disease","condition","predict"]):
                if st.session_state.result:
                    r2 = st.session_state.result
                    bot(f"The AI predicts <b>{r2['predicted_disease'].title()}</b> with "
                        f"<b>{r2['confidence']:.0f}%</b> confidence. Urgency: <b>{r2['urgency']}</b>.")
                else:
                    bot("No diagnosis yet — log your symptoms and press **LOG & ANALYZE**.")
            elif any(w in u for w in ["medicine","otc","drug","treatment","tablet"]):
                if st.session_state.result:
                    items = "\n".join(f"• {o}" for o in st.session_state.result["otc"][:4])
                    bot(f"<b>OTC guidance for your symptoms:</b><br>{items}")
                else:
                    bot("Please run an analysis first by logging symptoms and pressing **LOG & ANALYZE**.")
            elif any(w in u for w in ["duration","how long","long"]):
                if st.session_state.timeline:
                    durs = [e.get("duration","—") for e in st.session_state.timeline]
                    bot(f"Logged durations: <b>{', '.join(durs)}</b>. "
                        "Longer duration symptoms typically warrant a doctor visit.")
                else:
                    bot("No symptoms logged yet with duration info.")
            elif any(w in u for w in ["hello","hi","hey","good"]):
                bot(f"Hello Dr. <b>{st.session_state.username.title()}</b>! Ready to assist. "
                    "What would you like to know about your patient assessment?")
            else:
                bot(f"You asked: <i>{user_input}</i><br><br>"
                    "I can help with: risk assessment, diagnosis details, OTC guidance, "
                    "symptom summary, duration analysis, and next steps. "
                    "Try the quick reply buttons above for common queries.")
            st.rerun()


# ═══════════════════════════════════════════════════════════════════
#  RESULTS SECTION
# ═══════════════════════════════════════════════════════════════════
if st.session_state.result:
    result  = st.session_state.result
    risk_l  = result["risk_level"]
    disease = result["predicted_disease"]
    conf    = result["confidence"]

    rc, rh_cls, rb_cls, urg_cls, r_label = risk_meta(
        {"High":85,"Medium":55,"Low":20}[risk_l])
    r_emoji = {"High":"🔴","Medium":"🟡","Low":"🟢"}[risk_l]

    st.markdown("""<hr>
    <div class="sec-div">AI Analysis Results — Full Report</div>
    """, unsafe_allow_html=True)

    # Summary stats bar
    all_syms_used = total_symptoms()
    last_entry    = st.session_state.timeline[-1] if st.session_state.timeline else {}
    st.markdown(f"""
    <div style="display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:1.2rem;">
      <div style="background:rgba(0,60,140,.15);border:1px solid rgba(0,100,220,.15);
           border-radius:10px;padding:.5rem 1rem;text-align:center;flex:1;min-width:90px;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;color:#4da6ff;">{len(all_syms_used)}</div>
        <div style="font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;color:#1a4a7a;">Symptoms</div>
      </div>
      <div style="background:rgba(0,60,140,.15);border:1px solid rgba(0,100,220,.15);
           border-radius:10px;padding:.5rem 1rem;text-align:center;flex:1;min-width:90px;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;color:#4da6ff;">{len(st.session_state.timeline)}</div>
        <div style="font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;color:#1a4a7a;">Body Areas</div>
      </div>
      <div style="background:rgba(0,60,140,.15);border:1px solid rgba(0,100,220,.15);
           border-radius:10px;padding:.5rem 1rem;text-align:center;flex:1;min-width:90px;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;color:{rc};">{st.session_state.risk_score}%</div>
        <div style="font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;color:#1a4a7a;">Risk Score</div>
      </div>
      <div style="background:rgba(0,60,140,.15);border:1px solid rgba(0,100,220,.15);
           border-radius:10px;padding:.5rem 1rem;text-align:center;flex:1;min-width:90px;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;color:#4da6ff;">{last_entry.get('duration','—')}</div>
        <div style="font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;color:#1a4a7a;">Duration</div>
      </div>
      <div style="background:rgba(0,60,140,.15);border:1px solid rgba(0,100,220,.15);
           border-radius:10px;padding:.5rem 1rem;text-align:center;flex:1;min-width:90px;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;color:#4da6ff;">{last_entry.get('onset','—').split('(')[0][:8].strip()}</div>
        <div style="font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;color:#1a4a7a;">Onset</div>
      </div>
      <div style="background:rgba(0,60,140,.15);border:1px solid rgba(0,100,220,.15);
           border-radius:10px;padding:.5rem 1rem;text-align:center;flex:1;min-width:90px;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;color:#4da6ff;">{last_entry.get('sym_type','—').split('/')[0][:8].strip()}</div>
        <div style="font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;color:#1a4a7a;">Type</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown(f"""
    <div class="result-hero {rh_cls}">
      <div class="rh-top">
        <div class="rh-badge {rb_cls}">{r_emoji} {risk_l} Risk · AI Triage Result</div>
        <div class="rh-disease">{disease.title()}</div>
        <div class="rh-action">{result["recommendation"]}</div>
        <div class="rh-spec">👨‍⚕️ &nbsp;{result["specialist"]}</div>
        <div class="conf-box">
          <div class="conf-num">{conf:.0f}%</div>
          <div class="conf-lbl">Confidence</div>
        </div>
      </div>
      <div class="rh-bottom">
        <div class="rh-meta">Symptoms: <span>{len(result["matched_symptoms"])}</span></div>
        <div class="rh-meta">Urgency: <span>{result["urgency"]}</span></div>
        <div class="rh-meta">Duration: <span>{last_entry.get('duration','—')}</span></div>
        <div class="rh-meta">Onset: <span>{last_entry.get('onset','Gradual').split('(')[0].strip()}</span></div>
        <div class="rh-meta">Type: <span>{last_entry.get('sym_type','—').split('/')[0].strip()}</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Three result columns
    r1, r2, r3 = st.columns([1.1, 1, 1])

    with r1:
        chips = "".join(f"<span class='chip'>{s}</span>" for s in result["matched_symptoms"])
        bars  = ""
        for i, d in enumerate(result["top_diseases"]):
            pct = d["probability_pct"]
            t   = i == 0
            bars += f"""
            <div class="drow">
              <div class="dr-rank">#{i+1}</div>
              <div class="dr-name {'top' if t else ''}">{d['disease'].title()}</div>
              <div class="dr-track"><div class="dr-fill {'top' if t else ''}" style="width:{min(pct,100)}%"></div></div>
              <div class="dr-pct {'top' if t else ''}">{pct:.1f}%</div>
            </div>"""
        st.markdown(f"""
        <div class="ccard">
          <div class="ccard-header"><div class="ccard-icon ci-blue">🔬</div>
            <div class="ccard-title">Matched Symptoms</div></div>
          <div style="margin-bottom:1rem;">{chips}</div>
          <div class="ccard-header" style="margin-top:.8rem;">
            <div class="ccard-icon ci-purple">📊</div>
            <div class="ccard-title">Differential Diagnosis</div></div>
          {bars}
          <div style="font-size:.67rem;color:#1a3a5a;margin-top:.6rem;">
            * Relative probability within top-{st.session_state.top_n} differential.
          </div>
        </div>""", unsafe_allow_html=True)

    with r2:
        urg_cls2 = {"Emergency":"urg-e","Moderate":"urg-m","Non-urgent":"urg-n"}.get(result["urgency"],"urg-m")
        st.markdown(f"""
        <div class="ccard">
          <div class="ccard-header"><div class="ccard-icon ci-amber">⏱️</div>
            <div class="ccard-title">Urgency & Clinical Note</div></div>
          <div class="urg {urg_cls2}">{r_emoji} &nbsp;{result["urgency"]} — {risk_l} Risk</div>
          <div style="background:rgba(0,20,50,.5);border:1px solid rgba(0,80,160,.15);
               border-left:3px solid #0052cc;border-radius:0 10px 10px 0;
               padding:.9rem 1rem;font-size:.83rem;color:#6a9abf;line-height:1.7;">
            {result["note"]}
          </div>
          <div style="margin-top:1rem;padding-top:.8rem;border-top:1px solid rgba(0,80,160,.1);">
            <div style="font-size:.68rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;
                 color:#2a5a8a;margin-bottom:.6rem;">Symptom Summary</div>
            <div style="font-size:.78rem;color:#4a7a9b;line-height:1.8;">
              <b style="color:#6a9abf">Duration:</b> {last_entry.get('duration','—')}<br>
              <b style="color:#6a9abf">Onset:</b> {last_entry.get('onset','—')}<br>
              <b style="color:#6a9abf">Character:</b> {last_entry.get('sym_type','—')}<br>
              <b style="color:#6a9abf">Trigger:</b> {last_entry.get('trigger','—')}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with r3:
        otc_html = ""
        for i, item in enumerate(result["otc"], 1):
            tc = "otc-t w" if item.startswith(("⚠️","🚨")) else "otc-t"
            otc_html += f'<div class="otc-row"><span class="otc-n">{i:02d}</span><span class="{tc}">{item}</span></div>'
        st.markdown(f"""
        <div class="ccard">
          <div class="ccard-header"><div class="ccard-icon ci-green">💊</div>
            <div class="ccard-title">OTC & Home Management</div></div>
          {otc_html}
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disc">
      <span style="font-size:.95rem;flex-shrink:0">⚠️</span>
      <span><b>Clinical Disclaimer:</b> MedAI is for informational and decision-support purposes only.
      It does not replace a licensed medical professional. AI predictions are probabilistic.
      In emergencies call <b>112</b> (India) immediately.</span>
    </div>""", unsafe_allow_html=True)

# Logout
st.markdown("<hr style='margin:1.5rem 0;'>", unsafe_allow_html=True)
cl, _ = st.columns([1,6])
with cl:
    if st.button("⇤ LOGOUT", key="logout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
