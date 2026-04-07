import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PID-5 Clinical Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #F8FAFC; }
    .clinical-header {
        background: linear-gradient(135deg, #1B2A4A 0%, #0D7377 100%);
        padding: 2rem 2.5rem; border-radius: 12px;
        margin-bottom: 2rem;
    }
    .clinical-header h1 { font-size: 1.8rem; font-weight: 700; margin: 0; color: white; }
    .clinical-header p  { font-size: 0.95rem; opacity: 0.85; margin: 0.4rem 0 0 0; color: #A5C8D0; }
    .question-card {
        background: white; border: 1px solid #E2E8F0;
        border-radius: 10px; padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .question-number {
        font-size: 0.75rem; font-weight: 600; color: #0D7377;
        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem;
    }
    .question-text { font-size: 1rem; color: #1E293B; line-height: 1.5; }
    .result-high    { background: #FEE2E2; border-left: 4px solid #DC2626; padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0; }
    .result-flagged { background: #FEF3C7; border-left: 4px solid #D97706; padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0; }
    .result-low     { background: #F8FAFC; border-left: 4px solid #94A3B8; padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── PID-5 Structure ──────────────────────────────────────────────────────────
REVERSE_ITEMS = {7,30,35,58,87,90,96,97,98,131,142,155,164,177,210,215}

FACETS = {
    'Anhedonia':               [1,23,26,30,124,155,157,189],
    'Anxiousness':             [79,93,95,96,109,110,130,141,174],
    'Attention_Seeking':       [14,43,74,111,113,173,191,211],
    'Callousness':             [11,13,19,54,72,73,90,153,166,183,198,200,207,208],
    'Deceitfulness':           [41,53,56,76,126,134,142,206,214,218],
    'Depressivity':            [27,61,66,81,86,104,119,148,151,163,168,169,178,212],
    'Distractibility':         [6,29,47,68,88,118,132,144,199],
    'Eccentricity':            [5,21,24,25,33,52,55,70,71,152,172,185,205],
    'Emotional_Lability':      [18,62,102,122,138,165,181],
    'Grandiosity':             [40,65,114,179,187,197],
    'Hostility':               [28,32,38,85,92,116,158,170,188,216],
    'Impulsivity':             [4,16,17,22,58,204],
    'Intimacy_Avoidance':      [89,97,108,120,145,203],
    'Irresponsibility':        [31,129,156,160,171,201,210],
    'Manipulativeness':        [107,125,162,180,219],
    'Perceptual_Dysregulation':[36,37,42,44,59,77,83,154,192,193,213,217],
    'Perseveration':           [46,51,60,78,80,100,121,128,137],
    'Restricted_Affectivity':  [8,45,84,91,101,167,184],
    'Rigid_Perfectionism':     [34,49,105,115,123,135,140,176,196,220],
    'Risk_Taking':             [3,7,35,39,48,67,69,87,98,112,159,164,195,215],
    'Separation_Insecurity':   [12,50,57,64,127,149,175],
    'Submissiveness':          [9,15,63,202],
    'Suspiciousness':          [2,103,117,131,133,177,190],
    'Unusual_Beliefs':         [94,99,106,139,143,150,194,209],
    'Withdrawal':              [10,20,75,82,136,146,147,161,182,186],
}
FACET_NAMES = list(FACETS.keys())

DOMAINS = {
    'Negative_Affect': ['Emotional_Lability','Anxiousness','Separation_Insecurity'],
    'Detachment':      ['Withdrawal','Anhedonia','Intimacy_Avoidance'],
    'Antagonism':      ['Manipulativeness','Deceitfulness','Grandiosity'],
    'Disinhibition':   ['Irresponsibility','Impulsivity','Distractibility'],
    'Psychoticism':    ['Unusual_Beliefs','Eccentricity','Perceptual_Dysregulation'],
}

ICD_NAMES = {
    'F60.0':'Paranoid PD',     'F60.1':'Schizoid PD',
    'F60.2':'Antisocial PD',   'F60.3':'Borderline PD',
    'F60.4':'Histrionic PD',   'F60.5':'OCPD',
    'F60.6':'Avoidant PD',     'F60.7':'Dependent PD',
    'F60.81':'Narcissistic PD','F21':'Schizotypal',
}

THRESHOLD_FLAG = 0.35
THRESHOLD_HIGH = 0.60

QUESTIONS = {
    1:"I don't get as much pleasure out of things as others seem to.",
    2:"Plenty of people are out to get me.",
    3:"People would describe me as reckless.",
    4:"I feel like I act totally on impulse.",
    5:"I often have ideas that are too unusual to explain to anyone.",
    6:"I lose track of conversations because other things catch my attention.",
    7:"I avoid risky situations.",
    8:"When it comes to my emotions, people tell me I'm a 'cold fish'.",
    9:"I change what I do depending on what others want.",
    10:"I prefer not to get too close to people.",
    11:"I often get into physical fights.",
    12:"I dread being without someone to love me.",
    13:"Being rude and unfriendly is just a part of who I am.",
    14:"I do things to make sure people notice me.",
    15:"I usually do what others think I should do.",
    16:"I usually do things on impulse without thinking about what might happen as a result.",
    17:"Even though I know better, I can't stop making rash decisions.",
    18:"My emotions sometimes change for no good reason.",
    19:"I really don't care if I make other people suffer.",
    20:"I keep to myself.",
    21:"I often say things that others find odd or strange.",
    22:"I always do things on the spur of the moment.",
    23:"Nothing seems to interest me very much.",
    24:"Other people seem to think my behavior is weird.",
    25:"People have told me that I think about things in a really strange way.",
    26:"I almost never enjoy life.",
    27:"I often feel like nothing I do really matters.",
    28:"I snap at people when they do little things that irritate me.",
    29:"I can't concentrate on anything.",
    30:"I'm an energetic person.",
    31:"Others see me as irresponsible.",
    32:"I can be mean when I need to be.",
    33:"My thoughts often go off in odd or unusual directions.",
    34:"I've been told that I spend too much time making sure things are exactly in place.",
    35:"I avoid risky sports and activities.",
    36:"I can have trouble telling the difference between dreams and waking life.",
    37:"Sometimes I get this weird feeling that parts of my body feel like they're dead or not really me.",
    38:"I am easily angered.",
    39:"I have no limits when it comes to doing dangerous things.",
    40:"To be honest, I'm just more important than other people.",
    41:"I make up stories about things that happened that are totally untrue.",
    42:"People often talk about me doing things I don't remember at all.",
    43:"I do things so that people just have to admire me.",
    44:"It's weird, but sometimes ordinary objects seem to be a different shape than usual.",
    45:"I don't have very long-lasting emotional reactions to things.",
    46:"It is hard for me to stop an activity, even when it's time to do so.",
    47:"I'm not good at planning ahead.",
    48:"I do a lot of things that others consider risky.",
    49:"People tell me that I focus too much on minor details.",
    50:"I worry a lot about being alone.",
    51:"I've missed out on things because I was busy trying to get something I was doing exactly right.",
    52:"My thoughts often don't make sense to others.",
    53:"I often make up things about myself to help me get what I want.",
    54:"It doesn't really bother me to see other people get hurt.",
    55:"People often look at me as if I'd said something really weird.",
    56:"People don't realize that I'm flattering them to get something.",
    57:"I'd rather be in a bad relationship than be alone.",
    58:"I usually think before I act.",
    59:"I often see vivid dream-like images when I'm falling asleep or waking up.",
    60:"I keep approaching things the same way, even when it isn't working.",
    61:"I'm very dissatisfied with myself.",
    62:"I have much stronger emotional reactions than almost everyone else.",
    63:"I do what other people tell me to do.",
    64:"I can't stand being left alone, even for a few hours.",
    65:"I have outstanding qualities that few others possess.",
    66:"The future looks really hopeless to me.",
    67:"I like to take risks.",
    68:"I can't achieve goals because other things capture my attention.",
    69:"When I want to do something, I don't let the possibility that it might be risky stop me.",
    70:"Others seem to think I'm quite odd or unusual.",
    71:"My thoughts are strange and unpredictable.",
    72:"I don't care about other people's feelings.",
    73:"You need to step on some toes to get what you want in life.",
    74:"I love getting the attention of other people.",
    75:"I go out of my way to avoid any kind of group activity.",
    76:"I can be sneaky if it means getting what I want.",
    77:"Sometimes when I look at a familiar object, it's somehow like I'm seeing it for the first time.",
    78:"It is hard for me to shift from one activity to another.",
    79:"I worry a lot about terrible things that might happen.",
    80:"I have trouble changing how I'm doing something even if what I'm doing isn't going well.",
    81:"The world would be better off if I were dead.",
    82:"I keep my distance from people.",
    83:"I often can't control what I think about.",
    84:"I don't get emotional.",
    85:"I resent being told what to do, even by people in charge.",
    86:"I'm so ashamed by how I've let people down in lots of little ways.",
    87:"I avoid anything that might be even a little bit dangerous.",
    88:"I have trouble pursuing specific goals even for short periods of time.",
    89:"I prefer to keep romance out of my life.",
    90:"I would never harm another person.",
    91:"I don't show emotions strongly.",
    92:"I have a very short temper.",
    93:"I often worry that something bad will happen due to mistakes I made in the past.",
    94:"I have some unusual abilities, like sometimes knowing exactly what someone is thinking.",
    95:"I get very nervous when I think about the future.",
    96:"I rarely worry about things.",
    97:"I enjoy being in love.",
    98:"I prefer to play it safe rather than take unnecessary chances.",
    99:"I sometimes have heard things that others couldn't hear.",
    100:"I get fixated on certain things and can't stop.",
    101:"People tell me it's difficult to know what I'm feeling.",
    102:"I am a highly emotional person.",
    103:"Others would take advantage of me if they could.",
    104:"I often feel like a failure.",
    105:"If something I do isn't absolutely perfect, it's simply not acceptable.",
    106:"I often have unusual experiences, such as sensing the presence of someone who isn't actually there.",
    107:"I'm good at making people do what I want them to do.",
    108:"I break off relationships if they start to get close.",
    109:"I'm always worrying about something.",
    110:"I worry about almost everything.",
    111:"I like standing out in a crowd.",
    112:"I don't mind a little risk now and then.",
    113:"My behavior is often bold and grabs peoples' attention.",
    114:"I'm better than almost everyone else.",
    115:"People complain about my need to have everything all arranged.",
    116:"I always make sure I get back at people who wrong me.",
    117:"I'm always on my guard for someone trying to trick or harm me.",
    118:"I have trouble keeping my mind focused on what needs to be done.",
    119:"I talk about suicide a lot.",
    120:"I'm just not very interested in having sexual relationships.",
    121:"I get stuck on things a lot.",
    122:"I get emotional easily, often for very little reason.",
    123:"Even though it drives other people crazy, I insist on absolute perfection in everything I do.",
    124:"I almost never feel happy about my day-to-day activities.",
    125:"Sweet-talking others helps me get what I want.",
    126:"Sometimes you need to exaggerate to get ahead.",
    127:"I fear being alone in life more than anything else.",
    128:"I get stuck on one way of doing things, even when it's clear it won't work.",
    129:"I'm often pretty careless with my own and others' things.",
    130:"I am a very anxious person.",
    131:"People are basically trustworthy.",
    132:"I am easily distracted.",
    133:"It seems like I'm always getting a 'raw deal' from others.",
    134:"I don't hesitate to cheat if it gets me ahead.",
    135:"I check things several times to make sure they are perfect.",
    136:"I don't like spending time with others.",
    137:"I feel compelled to go on with things even when it makes little sense to do so.",
    138:"I never know where my emotions will go from moment to moment.",
    139:"I have seen things that weren't really there.",
    140:"It is important to me that things are done in a certain way.",
    141:"I always expect the worst to happen.",
    142:"I try to tell the truth even when it's hard.",
    143:"I believe that some people can move things with their minds.",
    144:"I can't focus on things for very long.",
    145:"I steer clear of romantic relationships.",
    146:"I'm not interested in making friends.",
    147:"I say as little as possible when dealing with people.",
    148:"I'm useless as a person.",
    149:"I'll do just about anything to keep someone from abandoning me.",
    150:"Sometimes I can influence other people just by sending my thoughts to them.",
    151:"Life looks pretty bleak to me.",
    152:"I think about things in odd ways that don't make sense to most people.",
    153:"I don't care if my actions hurt others.",
    154:"Sometimes I feel 'controlled' by thoughts that belong to someone else.",
    155:"I really live life to the fullest.",
    156:"I make promises that I don't really intend to keep.",
    157:"Nothing seems to make me feel good.",
    158:"I get irritated easily by all sorts of things.",
    159:"I do what I want regardless of how unsafe it might be.",
    160:"I often forget to pay my bills.",
    161:"I don't like to get too close to people.",
    162:"I'm good at conning people.",
    163:"Everything seems pointless to me.",
    164:"I never take risks.",
    165:"I get emotional over every little thing.",
    166:"It's no big deal if I hurt other peoples' feelings.",
    167:"I never show emotions to others.",
    168:"I often feel just miserable.",
    169:"I have no worth as a person.",
    170:"I am usually pretty hostile.",
    171:"I've skipped town to avoid responsibilities.",
    172:"I've been told more than once that I have a number of odd quirks or habits.",
    173:"I like being a person who gets noticed.",
    174:"I'm always fearful or on edge about bad things that might happen.",
    175:"I never want to be alone.",
    176:"I keep trying to make things perfect, even when I've gotten them as good as they're likely to get.",
    177:"I rarely feel that people I know are trying to take advantage of me.",
    178:"I know I'll commit suicide sooner or later.",
    179:"I've achieved far more than almost anyone I know.",
    180:"I can certainly turn on the charm if I need to get my way.",
    181:"My emotions are unpredictable.",
    182:"I don't deal with people unless I have to.",
    183:"I don't care about other peoples' problems.",
    184:"I don't react much to things that seem to make others emotional.",
    185:"I have several habits that others find eccentric or strange.",
    186:"I avoid social events.",
    187:"I deserve special treatment.",
    188:"It makes me really angry when people insult me in even a minor way.",
    189:"I rarely get enthusiastic about anything.",
    190:"I suspect that even my so-called 'friends' betray me a lot.",
    191:"I crave attention.",
    192:"Sometimes I think someone else is removing thoughts from my head.",
    193:"I have periods in which I feel disconnected from the world or from myself.",
    194:"I often see unusual connections between things that most people miss.",
    195:"I don't think about getting hurt when I'm doing things that might be dangerous.",
    196:"I simply won't put up with things being out of their proper places.",
    197:"I often have to deal with people who are less important than me.",
    198:"I sometimes hit people to remind them who's in charge.",
    199:"I get pulled off-task by even minor distractions.",
    200:"I enjoy making people in control look stupid.",
    201:"I just skip appointments or meetings if I'm not in the mood.",
    202:"I try to do what others want me to do.",
    203:"I prefer being alone to having a close romantic partner.",
    204:"I am very impulsive.",
    205:"I often have thoughts that make sense to me but that other people say are strange.",
    206:"I use people to get what I want.",
    207:"I don't see the point in feeling guilty about things I've done that have hurt other people.",
    208:"Most of the time I don't see the point in being friendly.",
    209:"I've had some really weird experiences that are very difficult to explain.",
    210:"I follow through on commitments.",
    211:"I like to draw attention to myself.",
    212:"I feel guilty much of the time.",
    213:"I often 'zone out' and then suddenly come to and realize that a lot of time has passed.",
    214:"Lying comes easily to me.",
    215:"I hate to take chances.",
    216:"I'm nasty and short to anybody who deserves it.",
    217:"Things around me often feel unreal, or more real than usual.",
    218:"I'll stretch the truth if it's to my advantage.",
    219:"It is easy for me to take advantage of others.",
    220:"I have a strict way of doing things.",
}

RESPONSE_OPTIONS = {
    0: "Very False or Often False",
    1: "Sometimes or Somewhat False",
    2: "Sometimes or Somewhat True",
    3: "Very True or Often True",
}

PAGES = 11
QUESTIONS_PER_PAGE = 20

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'pid5_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model_package = load_model()

# ─── Scoring Functions ────────────────────────────────────────────────────────
def score_pid5(raw_responses):
    scored = np.array(raw_responses, dtype=float)
    for item in REVERSE_ITEMS:
        scored[item - 1] = 3 - scored[item - 1]
    return {
        facet: round(float(np.mean([scored[i-1] for i in items])), 4)
        for facet, items in FACETS.items()
    }

def compute_domains(facet_scores):
    return {
        domain: round(float(np.mean([facet_scores[f] for f in facets])), 4)
        for domain, facets in DOMAINS.items()
    }

def classify_severity(domain_scores):
    overall = np.mean(list(domain_scores.values()))
    if overall < 0.70: return "None"
    if overall < 1.00: return "Mild"
    if overall < 1.50: return "Moderate"
    return "Severe"

def predict_icd10(facet_scores):
    model     = model_package["model"]
    icd_codes = model_package["icd_codes"]
    X = np.array([[facet_scores[f] for f in FACET_NAMES]])
    probs = np.column_stack([
        est.predict_proba(X)[:, 1] for est in model.estimators_
    ])[0]
    return {icd: round(float(p), 4) for icd, p in zip(icd_codes, probs)}

# ─── Session State ────────────────────────────────────────────────────────────
if 'page'           not in st.session_state: st.session_state.page = 'intro'
if 'current_q_page' not in st.session_state: st.session_state.current_q_page = 1
if 'responses'      not in st.session_state: st.session_state.responses = {}
if 'patient_name'   not in st.session_state: st.session_state.patient_name = ''
if 'result'         not in st.session_state: st.session_state.result = None

# ─── Header ───────────────────────────────────────────────────────────────────
def render_header(subtitle=""):
    st.markdown(f"""
    <div class="clinical-header">
        <h1>🧠 PID-5 Personality Assessment</h1>
        <p>Personality Inventory for DSM-5 | Clinical Decision Support System
        {f' &nbsp;|&nbsp; {subtitle}' if subtitle else ''}</p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INTRO PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == 'intro':
    render_header()
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### Welcome")
        st.markdown("""
        This questionnaire contains **220 statements** about how you would describe yourself.

        **Instructions:**
        - Select the response that **best describes you**
        - There are **no right or wrong answers**
        - Your responses are **confidential**
        - Please read each statement carefully

        **Response Scale:**
        - **0** = Very False or Often False
        - **1** = Sometimes or Somewhat False
        - **2** = Sometimes or Somewhat True
        - **3** = Very True or Often True
        """)
        st.markdown("---")
        st.markdown("#### Patient Information")
        col_a, col_b = st.columns(2)
        with col_a:
            name = st.text_input("Full Name / Patient ID", placeholder="e.g. John Smith")
        with col_b:
            st.text_input("Age", placeholder="e.g. 32")
        col_c, col_d = st.columns(2)
        with col_c:
            sex = st.selectbox("Sex", ["Select...", "Male", "Female", "Other", "Prefer not to say"])
        with col_d:
            st.text_input("Date", placeholder="e.g. 2026-04-07")
        st.markdown("")
        if st.button("▶  Begin Assessment", type="primary", use_container_width=True):
            if name and sex != "Select...":
                st.session_state.patient_name = name
                st.session_state.page = 'questionnaire'
                st.session_state.current_q_page = 1
                st.rerun()
            else:
                st.error("Please enter your name and select your sex to continue.")

    with col2:
        st.markdown("#### About This Assessment")
        st.info("""
        **PID-5 (Full Version)**

        Assesses **25 personality trait facets** across **5 domains**:

        🟡 **Negative Affect**
        Emotional Lability · Anxiousness · Separation Insecurity

        🟣 **Detachment**
        Withdrawal · Anhedonia · Intimacy Avoidance

        🔴 **Antagonism**
        Manipulativeness · Deceitfulness · Grandiosity

        🟠 **Disinhibition**
        Irresponsibility · Impulsivity · Distractibility

        🔵 **Psychoticism**
        Unusual Beliefs · Eccentricity · Perceptual Dysregulation
        """)
        st.warning("""
        ⚠️ **Clinical Use Only**

        Results must be reviewed by a licensed clinician.
        This tool does not provide a diagnosis.

        *Krueger et al. (2012) Psychol Med 42(9)*
        """)

# ══════════════════════════════════════════════════════════════════════════════
# QUESTIONNAIRE PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'questionnaire':
    qp = st.session_state.current_q_page
    total_answered = len(st.session_state.responses)
    progress = total_answered / 220

    render_header(f"Patient: {st.session_state.patient_name}")

    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
        <span style="font-size:0.85rem;color:#64748B;font-weight:500;">Page {qp} of {PAGES}</span>
        <span style="font-size:0.85rem;color:#0D7377;font-weight:600;">{total_answered} / 220 completed</span>
    </div>
    <div style="background:#E2E8F0;border-radius:999px;height:8px;margin-bottom:1.5rem;">
        <div style="background:linear-gradient(90deg,#0D7377,#14A5A3);border-radius:999px;height:8px;width:{progress*100:.1f}%;"></div>
    </div>
    """, unsafe_allow_html=True)

    start_q = (qp - 1) * QUESTIONS_PER_PAGE + 1
    end_q   = min(qp * QUESTIONS_PER_PAGE, 220)

    st.markdown(f"#### Questions {start_q}–{end_q}")
    st.markdown("*Select the response that best describes you.*")
    st.markdown("")

    for q_num in range(start_q, end_q + 1):
        current_val = st.session_state.responses.get(q_num, None)
        st.markdown(f"""
        <div class="question-card">
            <div class="question-number">Item {q_num}</div>
            <div class="question-text">{QUESTIONS[q_num]}</div>
        </div>
        """, unsafe_allow_html=True)
        options = ["Select a response..."] + [f"{k} — {v}" for k, v in RESPONSE_OPTIONS.items()]
        default_idx = 0 if current_val is None else current_val + 1
        selected = st.selectbox(
            f"q{q_num}", options=options,
            index=default_idx, label_visibility="collapsed",
            key=f"select_{q_num}"
        )
        if selected != "Select a response...":
            st.session_state.responses[q_num] = int(selected.split(" — ")[0])

    st.markdown("---")
    col_prev, col_info, col_next = st.columns([1, 2, 1])

    with col_prev:
        if qp > 1:
            if st.button("← Previous", use_container_width=True):
                st.session_state.current_q_page -= 1
                st.rerun()

    with col_info:
        page_answered = sum(1 for q in range(start_q, end_q+1) if q in st.session_state.responses)
        page_total    = end_q - start_q + 1
        if page_answered < page_total:
            st.warning(f"⚠ {page_total - page_answered} unanswered on this page")
        else:
            st.success(f"✓ All {page_total} questions answered")

    with col_next:
        if qp < PAGES:
            if st.button("Next →", type="primary", use_container_width=True):
                page_answered = sum(1 for q in range(start_q, end_q+1) if q in st.session_state.responses)
                if page_answered < (end_q - start_q + 1):
                    st.error("Please answer all questions before continuing.")
                else:
                    st.session_state.current_q_page += 1
                    st.rerun()
        else:
            if st.button("Submit Assessment →", type="primary", use_container_width=True):
                if len(st.session_state.responses) < 220:
                    st.error(f"{220 - len(st.session_state.responses)} questions remaining.")
                else:
                    with st.spinner("Analyzing responses..."):
                        responses_list = [st.session_state.responses[i] for i in range(1, 221)]
                        facet_scores   = score_pid5(responses_list)
                        domain_scores  = compute_domains(facet_scores)
                        severity       = classify_severity(domain_scores)
                        icd_probs      = predict_icd10(facet_scores)
                        flagged_facets = sorted(
                            [f for f, v in facet_scores.items() if v >= 1.5],
                            key=lambda f: -facet_scores[f]
                        )
                        icd_results = []
                        for code, prob in sorted(icd_probs.items(), key=lambda x: -x[1]):
                            risk_level = "HIGH" if prob >= THRESHOLD_HIGH else (
                                         "FLAGGED" if prob >= THRESHOLD_FLAG else "LOW")
                            icd_results.append({
                                'code': code, 'name': ICD_NAMES.get(code, code),
                                'probability': prob, 'risk_level': risk_level,
                                'flagged': prob >= THRESHOLD_FLAG,
                            })
                        top           = icd_results[0]
                        flagged_count = sum(1 for r in icd_results if r['flagged'])

                        if flagged_count == 0:
                            note = "No personality disorder risk flags detected. Routine follow-up recommended."
                        elif flagged_count == 1:
                            note = f"1 code flagged: {top['code']} ({top['name']}). Structured clinical interview recommended."
                        else:
                            codes = [r['code'] for r in icd_results if r['flagged']]
                            note  = (f"{flagged_count} codes flagged: {', '.join(codes)}. "
                                     f"Priority: {top['code']} (P={top['probability']:.2f}). "
                                     f"Comprehensive psychiatric evaluation recommended.")

                        st.session_state.result = {
                            'severity':       severity,
                            'icd10_results':  icd_results,
                            'top_diagnosis':  f"{top['code']} — {top['name']}",
                            'top_probability':top['probability'],
                            'flagged_count':  flagged_count,
                            'clinical_note':  note,
                            'facet_scores':   facet_scores,
                            'domain_scores':  domain_scores,
                            'flagged_facets': flagged_facets,
                        }
                        st.session_state.page = 'results'
                        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'results':
    result = st.session_state.result
    render_header(f"Results — {st.session_state.patient_name}")

    if result is None:
        st.error("No results found.")
        st.stop()

    sev = result['severity']
    sev_color = {'None':'#D1FAE5','Mild':'#DBEAFE','Moderate':'#FEF3C7','Severe':'#FEE2E2'}.get(sev,'#F8FAFC')
    sev_tc    = {'None':'#065F46','Mild':'#1E40AF','Moderate':'#92400E','Severe':'#7F1D1D'}.get(sev,'#1E293B')

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div style="background:{sev_color};padding:1.2rem;border-radius:10px;text-align:center;">
            <div style="font-size:0.8rem;color:{sev_tc};font-weight:600;text-transform:uppercase;">Severity</div>
            <div style="font-size:1.8rem;font-weight:700;color:{sev_tc};">{sev}</div></div>""",
            unsafe_allow_html=True)
    with c2:
        f = result['flagged_count']
        fc = '#FEE2E2' if f>2 else ('#FEF3C7' if f>0 else '#D1FAE5')
        ft = '#7F1D1D' if f>2 else ('#92400E' if f>0 else '#065F46')
        st.markdown(f"""<div style="background:{fc};padding:1.2rem;border-radius:10px;text-align:center;">
            <div style="font-size:0.8rem;color:{ft};font-weight:600;text-transform:uppercase;">Flagged</div>
            <div style="font-size:1.8rem;font-weight:700;color:{ft};">{f}</div></div>""",
            unsafe_allow_html=True)
    with c3:
        p = result['top_probability']
        pc = '#FEE2E2' if p>=0.6 else ('#FEF3C7' if p>=0.35 else '#D1FAE5')
        pt = '#7F1D1D' if p>=0.6 else ('#92400E' if p>=0.35 else '#065F46')
        st.markdown(f"""<div style="background:{pc};padding:1.2rem;border-radius:10px;text-align:center;">
            <div style="font-size:0.8rem;color:{pt};font-weight:600;text-transform:uppercase;">Top Probability</div>
            <div style="font-size:1.8rem;font-weight:700;color:{pt};">{p:.1%}</div></div>""",
            unsafe_allow_html=True)
    with c4:
        top_dx = result['top_diagnosis'].split('—')[0].strip()
        st.markdown(f"""<div style="background:#F8FAFC;padding:1.2rem;border-radius:10px;text-align:center;border:1px solid #E2E8F0;">
            <div style="font-size:0.8rem;color:#64748B;font-weight:600;text-transform:uppercase;">Primary Flag</div>
            <div style="font-size:1.4rem;font-weight:700;color:#1E293B;">{top_dx}</div></div>""",
            unsafe_allow_html=True)

    st.markdown("")
    st.info(f"📋 **Clinical Note:** {result['clinical_note']}")
    st.markdown("")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### ICD-10 Risk Probabilities")
        for r in result['icd10_results']:
            prob  = r['probability']
            level = r['risk_level']
            css   = 'result-high' if level=='HIGH' else ('result-flagged' if level=='FLAGGED' else 'result-low')
            icon  = '🔴' if level=='HIGH' else ('🟡' if level=='FLAGGED' else '⚪')
            bar_c = '#DC2626' if level=='HIGH' else ('#D97706' if level=='FLAGGED' else '#94A3B8')
            st.markdown(f"""
            <div class="{css}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div><span style="font-weight:600;">{icon} {r['code']}</span>
                    <span style="color:#64748B;font-size:0.85rem;margin-left:0.5rem;">{r['name']}</span></div>
                    <div style="font-weight:700;font-size:1.1rem;">{prob:.1%}</div>
                </div>
                <div style="background:#E2E8F0;border-radius:999px;height:5px;margin-top:0.5rem;">
                    <div style="background:{bar_c};border-radius:999px;height:5px;width:{prob*100:.1f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Domain Profile")
        domain_scores = result['domain_scores']
        domains = [d.replace('_', ' ') for d in domain_scores.keys()]
        values  = list(domain_scores.values())
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values+[values[0]], theta=domains+[domains[0]],
            fill='toself', fillcolor='rgba(13,115,119,0.15)',
            line=dict(color='#0D7377', width=2.5), name='Patient',
        ))
        fig.add_trace(go.Scatterpolar(
            r=[1.5]*(len(domains)+1), theta=domains+[domains[0]],
            line=dict(color='#DC2626', width=1.5, dash='dot'),
            name='Threshold (1.5)', fill='none',
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,3],
                       tickvals=[0.5,1.0,1.5,2.0,2.5], tickfont=dict(size=9))),
            showlegend=True, legend=dict(font=dict(size=10)),
            margin=dict(t=30,b=30,l=40,r=40), height=360,
            paper_bgcolor='white',
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Facet Scores")
    facet_scores   = result['facet_scores']
    flagged_facets = result.get('flagged_facets', [])
    facet_df = pd.DataFrame([
        {'Facet': f.replace('_',' '), 'Score': round(v,3),
         'Status': '⚠ Elevated' if f in flagged_facets else ''}
        for f, v in sorted(facet_scores.items(), key=lambda x: -x[1])
    ])

    def color_score(val):
        if val >= 2.0: return 'background-color:#FEE2E2;color:#7F1D1D;font-weight:bold'
        if val >= 1.5: return 'background-color:#FEF3C7;color:#92400E;font-weight:bold'
        if val >= 1.0: return 'background-color:#DBEAFE;color:#1E40AF'
        return 'color:#64748B'

    half = len(facet_df) // 2
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.dataframe(facet_df.iloc[:half].style.applymap(color_score, subset=['Score']),
                     hide_index=True, use_container_width=True, height=340)
    with col_f2:
        st.dataframe(facet_df.iloc[half:].style.applymap(color_score, subset=['Score']),
                     hide_index=True, use_container_width=True, height=340)

    st.markdown("""
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;
                padding:0.8rem 1.2rem;margin-top:1rem;font-size:0.82rem;color:#64748B;">
        🔴 ≥2.0 Significant &nbsp;|&nbsp; 🟡 ≥1.5 Clinical threshold &nbsp;|&nbsp;
        🔵 ≥1.0 Mild elevation &nbsp;|&nbsp;
        🔴 P≥0.60 HIGH risk &nbsp;|&nbsp; 🟡 P≥0.35 FLAGGED
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.warning("⚠️ For clinical support only. All results must be reviewed by a licensed clinician. "
               "*Krueger RF et al. (2012). APA DSM-5 AMPD (2013).*")
    st.markdown("")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("🔄 New Assessment", use_container_width=True):
            st.session_state.page = 'intro'
            st.session_state.responses = {}
            st.session_state.result = None
            st.session_state.current_q_page = 1
            st.rerun()
    with col_r2:
        if st.button("📋 Review Answers", use_container_width=True):
            st.session_state.page = 'questionnaire'
            st.session_state.current_q_page = 1
            st.rerun()
