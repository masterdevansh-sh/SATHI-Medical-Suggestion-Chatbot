# chat_cli_full_kind.py
# Extended wellness CLI — kinder replies and cleaner crisis handling.
# Save as chat_cli_full_kind.py and run with your venv python:
# & ".venv\Scripts\python.exe" "C:\path\to\chat_cli_full_kind.py"

import re
import inspect
import json
import os
import datetime
from gpt4all import GPT4All

# === CONFIG - adjust these to your environment ===
MODEL_DIR = r"C:\local-chatbot\Llama-3.2-3B-Instruct-Q4_0.gguf"
MODEL_FILE = "Llama-3.2-3B-Instruct-Q4_0.gguf"
SESSION_STORE_DIR = os.path.join(os.path.dirname(__file__), "sessions")
os.makedirs(SESSION_STORE_DIR, exist_ok=True)

# Crisis and safety (standard brief message)
CRISIS_RESPONSE = (
    "I’m really sorry you’re feeling that way. I can’t provide emergency help. "
    "If you are in immediate danger, please call your local emergency number now. "
    "If you are in the United States, call 988 for the Suicide & Crisis Lifeline. "
    "Would you like resources to contact someone right now?"
)

CRISIS_KEYWORDS = [
    "suicid", "kill myself", "end my life", "want to die",
    "hurt myself", "i will kill", "i'm going to kill", "harm myself",
    "want to end", "i can't go on", "no reason to live", "i'm done"
]

# === Model load (for conversational replies) ===
print("Loading model (this may take a minute)...")
try:
    model = GPT4All(model_name=MODEL_FILE, model_path=MODEL_DIR)
    print("Model loaded.\n")
except Exception as e:
    model = None
    print("Warning: model failed to load:", e)
    print("You can still use deterministic scoring and persistence features without LLM replies.\n")

# === Helper: generate param mapping for different gpt4all bindings ===
def get_allowed_generate_params(model_obj):
    try:
        sig = inspect.signature(model_obj.generate)
        return set(sig.parameters.keys())
    except Exception:
        return {"n_predict", "temp", "top_p", "top_k"}

ALLOWED_GEN_PARAMS = get_allowed_generate_params(model) if model else set()
PARAM_MAP = {"temperature": "temp", "max_tokens": "n_predict", "top_p": "top_p", "top_k": "top_k"}

def build_gen_kwargs(user_params):
    raw = {}
    for u_key, val in user_params.items():
        if u_key in PARAM_MAP:
            raw_name = PARAM_MAP[u_key]
            raw[raw_name] = val
    return {k: v for k, v in raw.items() if k in ALLOWED_GEN_PARAMS}

# === Session state & persistence ===
session = {
    "id": datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"),
    "profile": {
        "service_branch": None,
        "posting": None,
        "role": None,
        "age": None,
        "location": None
    },
    "self_report": {},
    "hr_metrics": {},
    "bio_metrics": {},
    "symptoms": {},
    "vitals": {},
    "history": [],
    "created_at": datetime.datetime.utcnow().isoformat()
}

def session_path(name=None):
    name = name or session["id"]
    return os.path.join(SESSION_STORE_DIR, f"session_{name}.json")

def save_session(name=None):
    path = session_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2)
    return path

def load_session(name):
    path = session_path(name)
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    session.update(data)
    return True

def list_sessions():
    files = [f for f in os.listdir(SESSION_STORE_DIR) if f.startswith("session_") and f.endswith(".json")]
    return files

# === Text analysis & scoring ===
neg_keywords = [
    "sad", "depress", "hopeless", "alone", "isolat", "worthless",
    "overwhelm", "stressed", "anxious", "panic", "tired", "exhaust",
    "insomnia", "no energy", "can't sleep"
]

def contains_crisis(text):
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in CRISIS_KEYWORDS)

def count_negatives(text):
    if not text:
        return 0
    t = text.lower()
    cnt = 0
    for k in neg_keywords:
        if k in t:
            cnt += 1
    for k in ["suicid", "kill myself", "end my life", "want to die", "i'm going to kill"]:
        if k in t:
            cnt += 3
    return cnt

def clamp(v, lo=0, hi=100):
    try:
        return max(lo, min(hi, int(round(v))))
    except:
        return lo

# Scoring components
def score_self_report(sr: dict):
    if "phq_score" in sr:
        try:
            phq = float(sr.get("phq_score", 0))
            s = 100 * (1 - min(27.0, phq) / 27.0)
            return clamp(s)
        except:
            pass
    text = sr.get("free_text", "")
    if text:
        neg = count_negatives(text)
        s = 100 - clamp(neg * 20, 0, 100)
        return clamp(s)
    return 75

def score_hr_metrics(hr: dict):
    base = 100.0
    leaves = float(hr.get("recent_leaves", 0))
    deployments = float(hr.get("deployment_count_12m", 0))
    hours = float(hr.get("avg_week_hours", 40))
    shifts = float(hr.get("shift_changes", 0))
    base -= clamp((leaves - 1) * 10, 0, 40)
    base -= clamp(deployments * 8, 0, 30)
    if hours > 50:
        base -= clamp((hours - 50) * 1.5, 0, 30)
    base -= clamp((shifts - 2) * 6, 0, 30)
    return clamp(base)

def score_biometrics(bio: dict):
    sleep = bio.get("sleep_hours_avg", None)
    hrv = bio.get("hrv", None)
    water = bio.get("water_intake_l_per_day", None)
    scores = []
    if sleep is not None:
        try:
            s = float(sleep)
            sc = clamp((s / 8.0) * 100)
            scores.append(sc)
        except:
            pass
    if hrv is not None:
        try:
            h = float(hrv)
            sc = clamp((h / 80.0) * 100)
            scores.append(sc)
        except:
            pass
    if water is not None:
        try:
            w = float(water)
            sc = clamp(min(1.5, w) / 1.5 * 100)
            scores.append(sc)
        except:
            pass
    if scores:
        return clamp(sum(scores) / len(scores))
    return 75

def score_chat(history_msgs):
    text = " ".join(history_msgs[-10:])
    neg = count_negatives(text)
    s = 100 - clamp(neg * 15, 0, 100)
    return clamp(s)

# Symptom urgency mapping
SYMPTOM_URGENCY = {
    "chestpain": 40,
    "shortness of breath": 40,
    "heart palpitations": 30,
    "blurred vision": 25,
    "fainting": 40,
    "severe dizziness": 30,
    "fever": 20,
    "nausea": 10,
    "extreme fatigue": 20,
    "severe headache": 20,
    "numbness": 30,
    "severe back pain": 15,
    "severe pain": 20,
    "suicidal": 100
}

def score_symptoms(symptoms: dict):
    """
    Returns (score:int, urgent_flags:list).
    score near 100 is good; lower means worse.
    """
    if not symptoms:
        return 80, []
    penalty = 0.0
    urgent_flags = []
    for name, sev in symptoms.items():
        if not name:
            continue
        nm = str(name).lower().strip()
        try:
            sev_val = float(sev)
        except:
            sev_val = 5.0
        base_pen = min(10.0, sev_val) * 3.0
        for k, w in SYMPTOM_URGENCY.items():
            if k in nm:
                base_pen += float(w) * (min(10.0, sev_val) / 10.0)
                if w >= 30:
                    urgent_flags.append(k)
        penalty += base_pen
    score_val = max(1, 100 - min(90, penalty))
    return clamp(score_val), list(set(urgent_flags))

# Composite wellness computation
def compute_wellness():
    s_self = score_self_report(session.get("self_report", {}))
    s_hr = score_hr_metrics(session.get("hr_metrics", {}))
    s_chat = score_chat(session.get("history", []))
    s_bio = score_biometrics(session.get("bio_metrics", {}))
    sym_result = score_symptoms(session.get("symptoms", {}))
    if isinstance(sym_result, tuple) and len(sym_result) == 2:
        s_symptoms, urgent = sym_result
    else:
        try:
            s_symptoms = int(sym_result)
        except:
            s_symptoms = 80
        urgent = []
    w_self, w_hr, w_chat, w_bio, w_sym = 0.30, 0.15, 0.15, 0.10, 0.30
    ws = round(w_self * s_self + w_hr * s_hr + w_chat * s_chat + w_bio * s_bio + w_sym * s_symptoms)
    crisis_flag = contains_crisis(" ".join(session.get("history", []))) or contains_crisis(session.get("self_report", {}).get("free_text", ""))
    if crisis_flag:
        ws = min(ws, 25)
        risk = "critical"
    else:
        if ws >= 75:
            risk = "low"
        elif ws >= 50:
            risk = "medium"
        elif ws >= 30:
            risk = "high"
        else:
            risk = "critical"
    flags = []
    if s_self < 50:
        flags.append("low_self_report")
    if s_hr < 50:
        flags.append("hr_stress_signals")
    if s_chat < 50:
        flags.append("negative_chat_signals")
    if s_bio < 50:
        flags.append("poor_biometrics")
    if urgent:
        flags.extend([f"urgent_{u}" for u in urgent])
    if crisis_flag and "self-harm language" not in flags:
        flags.append("self-harm language")
    assessment = {
        "wellness_score": max(1, min(100, ws)),
        "risk_level": risk,
        "flags": flags,
        "contributors": [
            {"feature": "self_report", "score": s_self, "weight": int(w_self*100)},
            {"feature": "hr_metrics", "score": s_hr, "weight": int(w_hr*100)},
            {"feature": "chat_history", "score": s_chat, "weight": int(w_chat*100)},
            {"feature": "biometrics", "score": s_bio, "weight": int(w_bio*100)},
            {"feature": "symptoms", "score": s_symptoms, "weight": int(w_sym*100)}
        ],
        "recommended_actions": [],
        "crisis": bool(crisis_flag),
        "crisis_response": CRISIS_RESPONSE if crisis_flag else None
    }
    if assessment["crisis"]:
        assessment["recommended_actions"] = [
            "Immediate human welfare contact and emergency services if in danger",
            "Offer immediate crisis resources"
        ]
    elif any(f.startswith("urgent_") for f in flags):
        assessment["recommended_actions"] = [
            "Arrange urgent clinical evaluation (same-day)",
            "Contact medical services or welfare officer for rapid assessment"
        ]
    else:
        if risk == "low":
            assessment["recommended_actions"] = ["Continue monitoring, encourage self-care and check-ins"]
        elif risk == "medium":
            assessment["recommended_actions"] = ["Schedule a welfare check-in with unit officer", "Encourage structured self-assessment"]
        elif risk == "high":
            assessment["recommended_actions"] = ["Arrange human contact within 24 hours", "Consider counseling and workload adjustments"]
        else:
            assessment["recommended_actions"] = ["Immediate welfare officer notification and human contact", "Consider emergency medical evaluation"]
    return assessment

# === Symptom -> simple rule-based diagnoses ===
DIAG_RULES = [
    (["chestpain", "shortness of breath", "heart palpitations"], "Possible cardiac issue - urgent evaluation"),
    (["fever", "nausea", "blurred vision"], "Possible infection - clinical assessment recommended"),
    (["severe headache", "blurred vision", "numbness"], "Neurological concern - seek medical review"),
    (["extreme fatigue", "poor appetite", "sleep problems"], "High stress or depression-like symptoms"),
    (["muscle", "joint", "back pain"], "Musculoskeletal strain or overuse"),
    (["anxiety", "panic", "insomnia", "anger"], "Acute stress or anxiety symptoms"),
]

def analyze_symptoms():
    text_symptoms = " ".join([f"{k} " for k in session.get("symptoms", {}).keys()]).lower()
    diagnoses = []
    for keys, diag in DIAG_RULES:
        if any(k in text_symptoms for k in keys):
            diagnoses.append(diag)
    if not diagnoses:
        diagnoses = ["No specific high-priority diagnosis from symptoms; recommend monitoring and human review."]
    return list(dict.fromkeys(diagnoses))

# === CLI helpers ===
def parse_kv_pairs(text):
    res = {}
    for m in re.finditer(r'(\w+)=(".*?"|\'.*?\'|[^ ]+)', text):
        key = m.group(1)
        val = m.group(2)
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        if re.fullmatch(r'-?\d+(\.\d+)?', str(val)):
            if '.' in str(val):
                val = float(val)
            else:
                val = int(val)
        res[key] = val
    return res

def parse_symptom_list(text):
    items = [s.strip() for s in re.split(r'[,\;]', text) if s.strip()]
    res = {}
    for it in items:
        if ":" in it:
            name, sev = it.split(":",1)
            try:
                sev = float(sev)
            except:
                sev = 5
        else:
            name = it
            sev = 5
        res[name.strip().lower()] = clamp(sev,1,10)
    return res

# === Model-driven reply with kinder prompt ===
OFF_TOPIC_KEYWORDS = [
    "pet", "dog", "cat", "shelter", "adopt", "adoption",
    "install", "activate", "python", "venv", "error",
    "how to", "terminal", "browser", "url", "website",
    "movie", "game", "music", "recipe", "cooking"
]

def is_non_wellness_query(user_text):
    if not user_text:
        return False
    t = user_text.lower()
    return any(k in t for k in OFF_TOPIC_KEYWORDS)

def run_model_reply(user_text, gen_params=None):
    # If no model, return brief empathetic fallback
    if not model:
        return "I'm here to listen — tell me more about what's been hardest for you lately."

    steer_back_question = "Before we continue, could you tell me briefly how you've been feeling lately (1–10)?"

    # Kinder system prompt: deeper empathy, reflect, brief coping strategy, one question
    system_prompt_base = (
        "You are a deeply empathetic counselor. Prioritize validating emotions and making the user feel heard. "
        "Structure: 1) One short reflection of the user's feeling (1-2 sentences). "
        "2) A concise normalizing statement (1 sentence). "
        "3) One practical, immediate coping suggestion the user can try in the next 5 minutes (1 sentence). "
        "4) One short open follow-up question to continue the conversation (single question). "
        "Do NOT provide medical or legal advice. Avoid role labels or internal notes, and do not output chain-of-thought. "
        "Keep each sentence short and kind."
    )

    if is_non_wellness_query(user_text):
        full_prompt = (
            f"{system_prompt_base}\n\nUser practical question:\n{user_text}\n\n"
            f"Assistant (first a 1-2 sentence factual answer if appropriate, then follow the empathic structure above):"
        )
    else:
        full_prompt = (
            f"{system_prompt_base}\n\nUser message about wellbeing:\n{user_text}\n\n"
            f"Assistant (follow the empathic structure above):"
        )

    defaults = {"temperature": 0.3, "max_tokens": 280, "top_p": 0.9, "top_k": 40}
    if gen_params:
        defaults.update(gen_params)
    gen_kwargs = build_gen_kwargs(defaults)

    try:
        out = model.generate(full_prompt, **gen_kwargs)
    except TypeError:
        out = model.generate(full_prompt)
    except Exception:
        return "Sorry — I couldn't generate a reply due to an error."

    if isinstance(out, (list, tuple)):
        out_text = "".join(map(str, out))
    else:
        out_text = str(out)

    out_text = re.sub(r'(?im)^\s*(user|assistant)\s*:\s*', '', out_text)
    out_text = re.sub(r'--\s*/\w+\s*=\s*\d+\s*', '', out_text)
    out_text = re.sub(r'\(Note:.*?\)', '', out_text, flags=re.I|re.S)
    out_text = out_text.strip()

    # Ensure steer-back question for off-topic queries
    if is_non_wellness_query(user_text):
        if steer_back_question.lower() not in out_text.lower():
            out_text = out_text + "\n\n" + steer_back_question
    else:
        if not re.search(r'\?', out_text):
            out_text = out_text + " Could you tell me briefly how you've been feeling (1-10)?"

    return out_text

# === CLI ===
def print_help():
    print("""
Commands:
  /help
  /profile branch="Army" posting="Camp X" role="medic" age=35 location="US"
  /survey phq=12 free="I feel..."
  /hr leaves=3 deployments=1 avg_hours=60 shift_changes=4
  /bio sleep=6.5 hrv=30 water=1.5 last_meal_hours=5
  /symptoms headache:5, chestpain:8, nausea
  /vitals temp=37.2 bp="120/80" hr=78
  /diagnose   -> rule-based considerations
  /recommend  -> recommended actions
  /score      -> show the full wellness assessment JSON
  /save name
  /load name
  /list
  /clear
  /clearall
  /history
  exit or quit
""")

print("Wellness CLI — kinder mode. Type /help for commands.\n")

while True:
    try:
        text = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        break
    if not text:
        continue
    if text.lower() in ("exit", "quit"):
        print("Goodbye.")
        break

    if text.startswith("/"):
        parts = text.split(" ", 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        if cmd == "/help":
            print_help(); continue
        if cmd == "/profile":
            kv = parse_kv_pairs(arg)
            for k, v in kv.items():
                session["profile"][k] = v
            print("Profile updated."); continue
        if cmd == "/survey":
            kv = parse_kv_pairs(arg)
            if "phq" in kv: session["self_report"]["phq_score"] = kv["phq"]
            if "free" in kv: session["self_report"]["free_text"] = kv["free"]
            print("Survey updated."); continue
        if cmd == "/hr":
            kv = parse_kv_pairs(arg)
            for k, v in kv.items():
                if k == "deployments":
                    session["hr_metrics"]["deployment_count_12m"] = v
                else:
                    session["hr_metrics"][k] = v
            print("HR updated."); continue
        if cmd == "/bio":
            kv = parse_kv_pairs(arg)
            if "sleep" in kv: session["bio_metrics"]["sleep_hours_avg"] = kv["sleep"]
            if "hrv" in kv: session["bio_metrics"]["hrv"] = kv["hrv"]
            if "water" in kv: session["bio_metrics"]["water_intake_l_per_day"] = kv["water"]
            if "last_meal_hours" in kv: session["bio_metrics"]["last_meal_hours_ago"] = kv["last_meal_hours"]
            print("Biometrics updated."); continue
        if cmd == "/symptoms":
            new = parse_symptom_list(arg)
            session["symptoms"].update(new)
            print("Symptoms updated."); continue
        if cmd == "/vitals":
            kv = parse_kv_pairs(arg)
            if "temp" in kv: session["vitals"]["temperature"] = kv["temp"]
            if "bp" in kv: session["vitals"]["blood_pressure"] = kv["bp"]
            if "hr" in kv: session["vitals"]["heart_rate"] = kv["hr"]
            print("Vitals updated."); continue
        if cmd == "/diagnose":
            diag = analyze_symptoms()
            print("Probable diagnoses / considerations:")
            for d in diag:
                print(" -", d)
            continue
        if cmd == "/recommend":
            assessment = compute_wellness()
            print("Recommendations:")
            for r in assessment["recommended_actions"]:
                print(" -", r)
            if assessment["crisis"]:
                print("\nCrisis response shown above when detected.")
            continue
        if cmd == "/score":
            assessment = compute_wellness()
            print(json.dumps(assessment, indent=2))
            continue
        if cmd == "/save":
            name = arg.strip() if arg.strip() else session["id"]
            path = save_session(name)
            print("Session saved to", path)
            continue
        if cmd == "/load":
            name = arg.strip()
            if not name:
                print("Provide a session name to load (see /list)."); continue
            ok = load_session(name)
            print("Loaded." if ok else "No such session:", name)
            continue
        if cmd == "/list":
            files = list_sessions()
            if not files:
                print("No saved sessions.")
            else:
                for f in files:
                    print(" -", f)
            continue
        if cmd == "/export":
            name = arg.strip()
            if not name:
                print("Please provide a filename to export to.")
            else:
                p = os.path.abspath(name)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(session, f, indent=2)
                print("Exported session to", p)
            continue
        if cmd == "/clear":
            session["history"].clear()
            session["self_report"].clear()
            session["hr_metrics"].clear()
            session["bio_metrics"].clear()
            session["symptoms"].clear()
            session["vitals"].clear()
            print("Cleared structured data (history preserved)."); continue
        if cmd == "/clearall":
            session["history"].clear()
            session["self_report"].clear()
            session["hr_metrics"].clear()
            session["bio_metrics"].clear()
            session["symptoms"].clear()
            session["vitals"].clear()
            session["profile"] = {k: None for k in session["profile"]}
            print("Cleared all session data."); continue
        if cmd == "/history":
            for i, h in enumerate(session["history"][-50:], 1):
                print(f"{i}: {h}")
            continue
        print("Unknown command. Type /help for commands.")
        continue

    # Normal chat input
    session["history"].append(text)

    # crisis detection (high-recall)
    if contains_crisis(text) or contains_crisis(session.get("self_report", {}).get("free_text", "")):
        # Immediate empathetic safety message — minimal UI output
        print("\nBot:", CRISIS_RESPONSE, "\n")
        # compute and save assessment internally but DO NOT print full JSON
        assessment = compute_wellness()
        try:
            save_session(session["id"])
        except Exception:
            pass
        # short confirmation only
        print("Your safety concern has been recorded. If you are in danger, please call emergency services now.\n")
        continue

    # model reply (kinder prompt)
    reply = run_model_reply(text) if model else "I'm here to listen — tell me more."

    # print reply
    print("\nBot:", reply, "\n")

    # deterministic assessment (concise summary only)
    assessment = compute_wellness()
    print("Wellness summary: score={}, risk={}".format(assessment["wellness_score"], assessment["risk_level"]))
    print("Top action:", assessment["recommended_actions"][0])
    # NOTE: full JSON is only shown when user runs /score
    try:
        save_session(session["id"])
    except Exception:
        pass
    print("")  # separator