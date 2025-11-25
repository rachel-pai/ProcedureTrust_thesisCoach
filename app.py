import os
import json
from pathlib import Path
from typing import List, Dict, Any, Literal

import numpy as np
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import csv
from datetime import datetime
import os
import re
import markdown as md  # 👈 新增

load_dotenv()
OPENAI_MODEL = os.getenv("EBCS_MODEL", "gpt-5-mini")
EMBED_MODEL = os.getenv("EBCS_EMBED_MODEL", "text-embedding-3-large")
# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
POLICY_COLLECTION = os.getenv("QDRANT_POLICY_COLLECTION", "policy_docs")
THESES_COLLECTION = os.getenv("QDRANT_THESES_COLLECTION", "thesis_segments")

from qdrant_client import QdrantClient, models

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
STAGES = ["proposal", "greenlight", "midterm", "final", "defense", "other"]
MODES = ["exploration", "precedents", "diagnose", "checklist", "plan_synthesis", "critique", "ethics", "defense_drill",
         "other"]
GAPS = ["content", "process", "knowledge", "precedent", "mixed", "unknown"]

client = OpenAI()

# 你可以根据需要改路径
# 你原来有：
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
PRE_FILE = LOG_DIR / "pre_survey_ebcs.csv"
POST_FILE = LOG_DIR / "post_survey_ebcs.csv"

# ✅ 新增：EBCS 版本的聊天轮次 & evidence 事件日志
CHAT_LOG_FILE = LOG_DIR / "chat_turns_ebcs.csv"
EVIDENCE_LOG_FILE = LOG_DIR / "evidence_events_ebcs.csv"


def append_csv_row(path: Path, fieldnames, row_dict):
    """Append one row to a CSV (create header if file doesn’t exist)."""
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


# -----------------------
# 工具函数
# -----------------------
import re
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ebcs")


def log_step(msg: str):
    logger.info(msg)
    print(msg)  # 本地跑的话直接在 terminal 里也能看到


def login_page():
    # If already logged in, skip login screen
    if st.session_state.get("user_id"):
        return

    st.write("Please enter your Participant ID and password.")

    # Use stable keys so reruns don't reset unexpectedly
    participant_id = st.text_input(
        "Participant ID",
        value="",
        key="login_participant_id",
    )
    password = st.text_input(
        "Password",
        value="",
        type="password",
        key="login_password",
    )

    if st.button("Start", type="primary"):
        pid = participant_id.strip()
        pw = password.strip()

        # Basic checks
        if not pid or not pw:
            st.warning("You must enter both Participant ID and password.")
            return

        if not pid.isdigit():
            st.error("Participant ID must be a number (e.g., 1, 2, 3…).")
            return

        # Simple mapping: {1: 'user1', 2: 'user2', ...}
        expected_pw = f"user{pid}"

        if pw != expected_pw:
            st.error("Incorrect Participant ID or password.")
            return

        # Successful login
        st.session_state["user_id"] = pid
        st.success(f"Welcome, user{pid}!")
        st.rerun()


# ------------------------
# Pre-survey dialog function
# ------------------------

LIKERT_SCALES = {
    # How often…?
    "freq": [
        (1, "Never"),
        (2, "Rarely"),
        (3, "Occasionally"),
        (4, "Sometimes"),
        (5, "Often"),
        (6, "Very often"),
        (7, "Always"),
    ],
    # Trust-related (before/after)
    "trust": [
        (1, "Not at all"),
        (2, "Very low"),
        (3, "Low"),
        (4, "Moderate"),
        (5, "High"),
        (6, "Very high"),
        (7, "Complete trust"),
    ],
    # Clarity (topic / RQ clarity)
    "clarity": [
        (1, "Not at all clear"),
        (2, "Slightly clear"),
        (3, "Somewhat clear"),
        (4, "Moderately clear"),
        (5, "Quite clear"),
        (6, "Very clear"),
        (7, "Extremely clear"),
    ],
    # Confidence about RQ / plan
    "confidence": [
        (1, "Not at all confident"),
        (2, "Slightly confident"),
        (3, "Somewhat confident"),
        (4, "Moderately confident"),
        (5, "Quite confident"),
        (6, "Very confident"),
        (7, "Extremely confident"),
    ],
    # Usefulness
    "usefulness": [
        (1, "Not useful at all"),
        (2, "Slightly useful"),
        (3, "Somewhat useful"),
        (4, "Moderately useful"),
        (5, "Quite useful"),
        (6, "Very useful"),
        (7, "Extremely useful"),
    ],
    # Procedural fairness
    "fairness": [
        (1, "Very unfair"),
        (2, "Unfair"),
        (3, "Somewhat unfair"),
        (4, "Neutral"),
        (5, "Somewhat fair"),
        (6, "Fair"),
        (7, "Very fair"),
    ],
    # Transparency
    "transparency": [
        (1, "Not transparent at all"),
        (2, "Slightly transparent"),
        (3, "Somewhat transparent"),
        (4, "Moderately transparent"),
        (5, "Quite transparent"),
        (6, "Very transparent"),
        (7, "Extremely transparent"),
    ],
    # Mental workload
    "load": [
        (1, "Very low workload"),
        (2, "Low"),
        (3, "Slightly low"),
        (4, "Moderate"),
        (5, "Slightly high"),
        (6, "High"),
        (7, "Extremely high workload"),
    ],
    # Satisfaction
    "satisfaction": [
        (1, "Very dissatisfied"),
        (2, "Dissatisfied"),
        (3, "Slightly dissatisfied"),
        (4, "Neutral"),
        (5, "Slightly satisfied"),
        (6, "Satisfied"),
        (7, "Very satisfied"),
    ],
    # Generic fallback
    "generic": [
        (1, "Very low"),
        (2, "Low"),
        (3, "Slightly low"),
        (4, "Neutral"),
        (5, "Slightly high"),
        (6, "High"),
        (7, "Very high"),
    ],
}


# def likert_radio(label: str, default_value: int, key: str, scale: str = "generic") -> int:
def likert_radio(label: str, key: str, scale: str = "generic") -> int:
    """
    Show a 1–7 Likert scale as a HORIZONTAL radio with text labels only.
    Returns the numeric value (1–7).
    """
    items = LIKERT_SCALES.get(scale, LIKERT_SCALES["generic"])

    # Only the text is shown to the user
    options = [txt for (_v, txt) in items]

    # Default index based on stored numeric value
    # dv = int(default_value)
    # default_index = 3  # fallback
    # for i, (v, _txt) in enumerate(items):
    #     if v == dv:
    #         default_index = i
    #         break

    choice = st.radio(
        label,
        options,
        # index=default_index,
        index=None,
        key=key,
        horizontal=True,
    )

    # Map selected text back to numeric value
    for v, txt in items:
        if txt == choice:
            return v
    return items[3][0]  # fallback: middle value


@st.dialog("Before you start: Pre-Survey", dismissible=False, on_dismiss="rerun")
def pre_survey_dialog():
    st.write("Please complete this short pre-survey before using the Thesis Coach system.")

    user_id = st.session_state.get("user_id", None)
    if not user_id:
        st.error("You are not logged in. Please return to the login page.")
        return

    # ——— Initialize pre_tmp (first time only) ———
    if "pre_tmp" not in st.session_state or not isinstance(st.session_state.pre_tmp, dict):
        st.session_state.pre_tmp = {
            # baseline 四题
            "prior_exp_llm": 3,
            "prior_trust": 4,
            "topic_clarity": 3,
            "rq_confidence": 3,
            "open_expectations": "",
            # 新增：背景 + 自我效能 + 程序化偏好
            "stage": "",
            "domain_short": "",
            "rubric_familiarity": 2,
            "rq_self_efficacy": 3,
            "method_self_efficacy": 3,
            "rubric_eval_knowledge": 3,
            "procedural_preference": 4,
            "procedural_acceptance": 4,
        }

    pre_tmp = st.session_state.pre_tmp

    # =========================
    #  A. 背景信息
    # =========================
    st.subheader("A. Your current graduation context")

    stage_options = [
        "Not started yet",
        "Exploring topics / ideas",
        "Thesis proposal or pre-proposal",
        "Preparing for green-light",
        "Midterm phase",
        "Final writing / finishing phase",
        "Other / not sure",
    ]
    stage_default = pre_tmp.get("stage", "") or stage_options[0]
    stage = st.selectbox(
        "Which stage are you currently in with your graduation project?",
        options=stage_options,
        index=stage_options.index(stage_default) if stage_default in stage_options else 0,
        key="pre_stage",
    )
    st.session_state.pre_tmp["stage"] = stage

    domain_default = pre_tmp.get("domain_short", "")
    domain_short = st.text_input(
        "In one or two short phrases, how would you describe your graduation topic or domain? "
        "(e.g., hospital robot collaboration, warehouse teamwork, etc.)",
        value=domain_default,
        key="pre_domain_short",
    )
    st.session_state.pre_tmp["domain_short"] = domain_short

    rubric_scale = {
        1: "I have never seen the official IDE thesis rubrics / checklists",
        2: "I have heard of them but never looked at them in detail",
        3: "I have looked at them a few times",
        4: "I often refer to them when working on my thesis",
    }
    rubric_familiarity = st.radio(
        "Before using this system, how familiar are you with the official IDE graduation rubrics or stage checklists?",
        options=list(rubric_scale.keys()),
        # index=max(0, min(3, rubric_familiarity_default - 1)),
        index=None,
        format_func=lambda x: rubric_scale[x],
        horizontal=True,
        key="pre_rubric_familiarity",
    )
    st.session_state.pre_tmp["rubric_familiarity"] = rubric_familiarity

    # =========================
    #  B. 之前对 LLM 的经验 & 信任（baseline 部分）
    # =========================
    st.subheader("B. Prior experience with AI tools")

    prior_exp_llm = likert_radio(
        "How often have you previously used large language models (e.g., ChatGPT) for study or academic work?",
        # prior_exp_llm_default,
        key="pre_prior_exp_llm",
        scale="freq",
    )
    st.session_state.pre_tmp["prior_exp_llm"] = prior_exp_llm
    prior_trust = likert_radio(
        "Before using this system, how much do you trust AI-based feedback or supervision tools?",
        # prior_trust_default,
        key="pre_prior_trust",
        scale="trust",
    )
    st.session_state.pre_tmp["prior_trust"] = prior_trust

    # =========================
    #  C. 课题清晰度 & 自我效能（baseline + 扩展）
    # =========================
    st.subheader("C. Your thesis clarity and confidence")
    topic_clarity = likert_radio(
        "How clear are you about your graduation project topic or research direction?",
        # topic_clarity_default,
        key="pre_topic_clarity",
        scale="clarity",
    )
    st.session_state.pre_tmp["topic_clarity"] = topic_clarity
    rq_confidence = likert_radio(
        "How confident are you right now about your research question or thesis plan?",
        # rq_confidence_default,
        key="pre_rq_confidence",
        scale="confidence",
    )
    st.session_state.pre_tmp["rq_confidence"] = rq_confidence

    rq_self_efficacy = likert_radio(
        "Without any special tools, how confident are you that you can formulate a good research question (RQ)?",
        # rq_self_efficacy_default,
        key="pre_rq_self_efficacy",
        scale="confidence",
    )
    st.session_state.pre_tmp["rq_self_efficacy"] = rq_self_efficacy
    method_self_efficacy = likert_radio(
        "Without any special tools, how confident are you in choosing suitable methods and measurements to support your RQ?",
        # method_self_efficacy_default,
        key="pre_method_self_efficacy",
        scale="confidence",
    )
    st.session_state.pre_tmp["method_self_efficacy"] = method_self_efficacy

    rubric_eval_knowledge_default = int(pre_tmp.get("rubric_eval_knowledge", 3))
    rubric_eval_knowledge = likert_radio(
        "How well do you think you understand how IDE assessors will judge whether a thesis plan is ‘ready’ based on the official criteria?",
        # rubric_eval_knowledge_default,
        key="pre_rubric_eval_knowledge",
        scale="generic",
    )
    st.session_state.pre_tmp["rubric_eval_knowledge"] = rubric_eval_knowledge

    # =========================
    #  D. 对“按流程来”的态度（procedural orientation）
    # =========================
    st.subheader("D. Your attitude towards structured processes")

    procedural_preference_default = int(pre_tmp.get("procedural_preference", 4))
    procedural_preference = likert_radio(
        "In your study or projects, how much do you prefer having clear steps and checkpoints, instead of full freedom?",
        # procedural_preference_default,
        key="pre_procedural_preference",
        scale="generic",
    )
    st.session_state.pre_tmp["procedural_preference"] = procedural_preference

    procedural_acceptance_default = int(pre_tmp.get("procedural_acceptance", 4))
    procedural_acceptance = likert_radio(
        "If a digital tool requires you to follow a certain process (e.g., read rubrics, check evidence) before continuing, how acceptable is that to you?",
        # procedural_acceptance_default,
        key="pre_procedural_acceptance",
        scale="generic",
    )
    st.session_state.pre_tmp["procedural_acceptance"] = procedural_acceptance

    # =========================
    #  E. 开放性问题（baseline）
    # =========================
    st.subheader("E. Expectations for the Thesis Coach")

    open_expectations_default = pre_tmp.get("open_expectations", "")
    open_expectations = st.text_area(
        "What do you expect the Thesis Coach to help you with? (optional)",
        open_expectations_default,
        key="pre_open_expectations",
    )
    st.session_state.pre_tmp["open_expectations"] = open_expectations

    # —— Submit ——
    if st.button("Submit", type="primary"):
        missing = []

        if None in (stage, domain_short, rubric_familiarity, prior_exp_llm, prior_trust, topic_clarity, rq_confidence,
                    rq_self_efficacy, method_self_efficacy, rubric_eval_knowledge, procedural_preference,
                    procedural_acceptance):
            missing.append("Some required fields were not answered.")

        if missing:
            st.error("You must answer all required questions:\n- " + "\n- ".join(missing))
            st.stop()

        st.session_state["pre_survey"] = st.session_state.pre_tmp.copy()
        st.session_state["pre_survey_done"] = True

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            # 背景
            "stage": stage,
            "domain_short": domain_short,
            "rubric_familiarity": rubric_familiarity,
            # baseline LLM & trust
            "prior_exp_llm": prior_exp_llm,
            "prior_trust": prior_trust,
            # baseline clarity & confidence
            "topic_clarity": topic_clarity,
            "rq_confidence": rq_confidence,
            # 自我效能 & 程序化偏好
            "rq_self_efficacy": rq_self_efficacy,
            "method_self_efficacy": method_self_efficacy,
            "rubric_eval_knowledge": rubric_eval_knowledge,
            "procedural_preference": procedural_preference,
            "procedural_acceptance": procedural_acceptance,
            # 开放题
            "open_expectations": open_expectations,
        }
        append_csv_row(PRE_FILE, fieldnames=list(row.keys()), row_dict=row)

        st.success("Thank you! You may now use the system.")
        st.rerun()


def maybe_show_pre_survey():
    # 如果已经填过，就不再弹
    if st.session_state.get("pre_survey_done"):
        return
    # 第一次进入主界面时自动弹
    pre_survey_dialog()


# ------------------------
# Post Survey Dialog
# ------------------------

@st.dialog("Post-Survey", dismissible=False, on_dismiss="rerun")
def post_survey_dialog():
    st.write("Please complete this short post-survey after using the Thesis Coach system.")

    user_id = st.session_state.get("user_id", None)
    if not user_id:
        st.error("You are not logged in. Please return to the login page.")
        return

    # ——— Initialize post_tmp (first time only) ———
    if "post_tmp" not in st.session_state or not isinstance(st.session_state.post_tmp, dict):
        st.session_state.post_tmp = {
            # baseline
            "perceived_usefulness": 4,
            "perceived_procedural_fairness": 4,
            "perceived_transparency": 4,
            "trust_after": 4,
            "clarity_improved": 4,
            "cognitive_load": 4,
            "satisfaction": 4,
            "open_feedback": "",
            # 扩展：procedural trust / evidence use / calibration / usability / UI
            "procedural_rules_clarity": 4,
            "procedural_predictability": 4,
            "procedural_voice": 4,
            "evidence_engagement": 4,
            "evidence_cross_check": 4,
            "safety_support": 4,
            "trust_double_check": 4,
            "overtrust_concern": 4,
            "usability_ease": 4,
            "helpful_elements": [],  # list in state
        }

    post_tmp = st.session_state.post_tmp

    # ===== A. Overall experience (含 baseline 部分) =====
    st.subheader("A. Overall experience")

    perceived_usefulness = likert_radio(
        "Overall, how useful was the Thesis Coach for your current thesis task?",
        # int(post_tmp.get("perceived_usefulness", 4)),
        key="post_perceived_usefulness",
        scale="usefulness",
    )
    st.session_state.post_tmp["perceived_usefulness"] = perceived_usefulness

    clarity_improved = likert_radio(
        "Did the session help you become clearer about your thesis problem, RQ, or next steps?",
        # int(post_tmp.get("clarity_improved", 4)),
        key="post_clarity_improved",
        scale="clarity",
    )
    st.session_state.post_tmp["clarity_improved"] = clarity_improved

    trust_after = likert_radio(
        "After using this system, how much do you trust its feedback and guidance?",
        # int(post_tmp.get("trust_after", 4)),
        key="post_trust_after",
        scale="trust",
    )
    st.session_state.post_tmp["trust_after"] = trust_after

    satisfaction = likert_radio(
        "Overall, how satisfied are you with this interaction with the Thesis Coach?",
        # int(post_tmp.get("satisfaction", 4)),
        key="post_satisfaction",
        scale="satisfaction",
    )
    st.session_state.post_tmp["satisfaction"] = satisfaction

    usability_ease = likert_radio(
        "Overall, how easy or difficult was it to use the system to complete this task?",
        # int(post_tmp.get("usability_ease", 4)),
        key="post_usability_ease",
        scale="generic",
    )
    st.session_state.post_tmp["usability_ease"] = usability_ease

    cognitive_load = likert_radio(
        "How mentally demanding did you find the interaction with the system?",
        # int(post_tmp.get("cognitive_load", 4)),
        key="post_cognitive_load",
        scale="load",
    )
    st.session_state.post_tmp["cognitive_load"] = cognitive_load

    # ===== B. Procedural fairness & transparency（baseline + 拆分）=====
    st.subheader("B. Procedural fairness and transparency")

    perceived_procedural_fairness = likert_radio(
        "Did the process of getting feedback from the system feel systematic and fair (e.g., based on clear criteria rather than arbitrary answers)?",
        # int(post_tmp.get("perceived_procedural_fairness", 4)),
        key="post_perceived_procedural_fairness",
        scale="fairness",
    )
    st.session_state.post_tmp["perceived_procedural_fairness"] = perceived_procedural_fairness

    perceived_transparency = likert_radio(
        "How transparent did the system feel about *why* it gave particular suggestions (e.g., showing rubrics or precedents)?",
        # int(post_tmp.get("perceived_transparency", 4)),
        key="post_perceived_transparency",
        scale="transparency",
    )
    st.session_state.post_tmp["perceived_transparency"] = perceived_transparency

    procedural_rules_clarity = likert_radio(
        "During this session, how clear did it feel that the system was following a consistent set of rules or criteria?",
        # int(post_tmp.get("procedural_rules_clarity", 4)),
        key="post_procedural_rules_clarity",
        scale="generic",
    )
    st.session_state.post_tmp["procedural_rules_clarity"] = procedural_rules_clarity

    procedural_predictability = likert_radio(
        "How predictable did the system’s next steps feel (e.g., what it would ask you to do next)?",
        # int(post_tmp.get("procedural_predictability", 4)),
        key="post_procedural_predictability",
        scale="generic",
    )
    st.session_state.post_tmp["procedural_predictability"] = procedural_predictability

    procedural_voice = likert_radio(
        "When you did not fully agree with the system’s suggestions, did you still feel you had room to express your own ideas or choose another path?",
        # int(post_tmp.get("procedural_voice", 4)),
        key="post_procedural_voice",
        scale="generic",
    )
    st.session_state.post_tmp["procedural_voice"] = procedural_voice

    # ===== C. Evidence use & perceived safety =====
    st.subheader("C. Evidence use and perceived safety")

    evidence_engagement = likert_radio(
        "This system encouraged me to actually open and read the sources or snippets it provided (instead of just trusting the answer).",
        # int(post_tmp.get("evidence_engagement", 4)),
        key="post_evidence_engagement",
        scale="generic",
    )
    st.session_state.post_tmp["evidence_engagement"] = evidence_engagement

    evidence_cross_check = likert_radio(
        "Before making decisions, I usually checked whether at least one or two snippets really supported the system’s suggestions.",
        # int(post_tmp.get("evidence_cross_check", 4)),
        key="post_evidence_cross_check",
        scale="generic",
    )
    st.session_state.post_tmp["evidence_cross_check"] = evidence_cross_check

    safety_support = likert_radio(
        "In this task, I felt that the system helped me avoid decisions that might be risky or weak for my thesis.",
        # int(post_tmp.get("safety_support", 4)),
        key="post_safety_support",
        scale="generic",
    )
    st.session_state.post_tmp["safety_support"] = safety_support

    # ===== D. Trust calibration =====
    st.subheader("D. Trust calibration")

    trust_double_check = likert_radio(
        "Even when the system sounded confident, I still used my own judgement or other information to double-check important suggestions.",
        # int(post_tmp.get("trust_double_check", 4)),
        key="post_trust_double_check",
        scale="generic",
    )
    st.session_state.post_tmp["trust_double_check"] = trust_double_check

    overtrust_concern = likert_radio(
        "At some moments in this task, I felt that I might be relying too much on the system.",
        # int(post_tmp.get("overtrust_concern", 4)),
        key="post_overtrust_concern",
        scale="generic",
    )
    st.session_state.post_tmp["overtrust_concern"] = overtrust_concern

    # ===== E. UI elements + open feedback =====
    st.subheader("E. Interface elements")

    helpful_elements_options = [
        "The general chat interface",
        "The structured survey dialogs (pre / post questions themselves)",
        "The separate sources / snippet buttons",
        "The right-hand snippet panel with raw text",
        "Seeing titles and similarity scores of retrieved snippets",
        "Other elements (please describe in the text box below)",
        "None of the above were particularly helpful",
    ]
    # 在 state 里保存为 list，避免你之前遇到的“要点两次才能删”的问题
    helpful_default_list = post_tmp.get("helpful_elements", [])
    if not isinstance(helpful_default_list, list):
        helpful_default_list = []

    helpful_elements_selected = st.multiselect(
        "Which interface elements of the Thesis Coach did you personally find especially helpful in this session? (you can select multiple)",
        options=helpful_elements_options,
        default=[opt for opt in helpful_default_list if opt in helpful_elements_options],
        key="post_helpful_elements",
    )
    st.session_state.post_tmp["helpful_elements"] = helpful_elements_selected

    open_feedback_default = post_tmp.get("open_feedback", "")
    open_feedback = st.text_area(
        "If you have any comments about what worked well or what felt problematic (e.g., fairness, clarity, missing support), please write them here. (optional)",
        open_feedback_default,
        key="post_open_feedback",
    )
    st.session_state.post_tmp["open_feedback"] = open_feedback

    # —— Submit ——
    if st.button("Submit", type="primary"):
        missing = []

        if None in (perceived_usefulness, perceived_procedural_fairness, perceived_transparency,
                    trust_after, clarity_improved, cognitive_load, satisfaction,
                    usability_ease, procedural_rules_clarity, procedural_predictability, procedural_voice,
                    evidence_engagement, evidence_cross_check, safety_support, trust_double_check, overtrust_concern):
            missing.append("Some required fields were not answered.")

        if missing:
            st.error("You must answer all required questions:\n- " + "\n- ".join(missing))
            st.stop()

        st.session_state["post_survey"] = st.session_state.post_tmp.copy()
        st.session_state["post_survey_done"] = True

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            # baseline
            "perceived_usefulness": perceived_usefulness,
            "perceived_procedural_fairness": perceived_procedural_fairness,
            "perceived_transparency": perceived_transparency,
            "trust_after": trust_after,
            "clarity_improved": clarity_improved,
            "cognitive_load": cognitive_load,
            "satisfaction": satisfaction,
            "open_feedback": open_feedback,
            # 扩展
            "usability_ease": usability_ease,
            "procedural_rules_clarity": procedural_rules_clarity,
            "procedural_predictability": procedural_predictability,
            "procedural_voice": procedural_voice,
            "evidence_engagement": evidence_engagement,
            "evidence_cross_check": evidence_cross_check,
            "safety_support": safety_support,
            "trust_double_check": trust_double_check,
            "overtrust_concern": overtrust_concern,
            "helpful_elements": "; ".join(helpful_elements_selected),
        }
        append_csv_row(POST_FILE, fieldnames=list(row.keys()), row_dict=row)

        st.success("Thank you for your feedback!")
        st.rerun()


def show_intro_banner():
    """Show a one-time intro window explaining how to ask the coach."""
    if "show_intro" not in st.session_state:
        st.session_state.show_intro = True

    if not st.session_state.show_intro:
        return

    with st.container():
        st.markdown(
            """
<div style="
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    border: 1px solid #4b5563;
">
<h3 style="margin-top:0;">How to ask the Thesis Coach</h3>

To get **fast, evidence-based answers** (without extra clarification), try to include:

1. **Domain / topic** – what you’re working on (system, context, problem).
2. **Users / stakeholders** – who it is for.
3. **Key metrics** – 1–2 things you want to improve or evaluate (e.g., usability, adoption, workload).
4. **Stage** *(optional but helpful)* – proposal / greenlight / midterm / final / defense.
5. **Draft / method** *(when relevant)* – say if you already have an RQ draft, outline, or planned method.
---
#### 👍 Good question examples
**Example 1 – proposal / exploration**
> I’m designing an AR interface to guide warehouse pickers.
> Main users are novice pickers in large e-commerce warehouses.
> I mainly care about task completion time and error rate.
> I’m in the proposal stage with a rough RQ draft.
> Can you help me improve the research question and check if it fits the IDE proposal rubric?

**Example 2 – greenlight / checklist**
> I’m preparing my green-light plan for a mental-health chatbot for university students.
> Users: Dutch master’s students experiencing study stress.
> Metrics: engagement (return visits) and perceived support.
> I have a draft method (diary study + interviews).
> Can you use the green-light checklist to tell me what is missing?

**Example 3 – precedents**
> I’m working on a decision-support dashboard for ICU nurses.
> Users are experienced nurses; I want to evaluate situation awareness and workload.
> I plan a within-subjects lab study.
> Can you show me IDE thesis precedents with similar methods and metrics?
---
#### 👎 Bad question examples (and why they’re bad)
**Bad 1** ❌ No domain, no users, no metrics → the coach can only ask you a follow-up.
> I need help starting my thesis.

**Bad 2** ❌ Domain too vague, no users, no metrics, no stage.
> I’m doing something with sustainability. What should I do next?

**Bad 3** ❌ No context: we don’t know which stage you’re in, what your topic is, or what “fix” means.
> Can you fix my thesis?
-----
You can think of it as a template:
> “I’m working on **[domain]** for **[users]**.
> I mainly care about **[metric 1]** and **[metric 2]**, and I’m at the **[stage]** stage.
> I currently have **[draft / method / no draft yet]**.
> **Can you help me with [X]?**”
</div>
            """,
            unsafe_allow_html=True,
        )
        # cols = st.columns([1, 0.2])
        # with cols[0]:
        if st.button("Got it", key="intro_dismiss", use_container_width=True):
            st.session_state.show_intro = False
            st.rerun()


def extract_json(text: str):
    # 去掉可能的 ```json 或 ``` 包装
    text = re.sub(r"^```(?:json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())

    # 截取第一个 { 到最后一个 }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        return json.loads(json_str)
    else:
        raise ValueError("No JSON found")


def embed_text(text: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
    )
    log_step("Step 1 done: embedding received.")
    return resp.data[0].embedding


def cosine_sim_matrix(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    denom = (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-9)
    return (matrix @ query) / denom


# ========= Qdrant Repositories =========
class PolicyItemFromPayload:
    def __init__(self, payload: dict):
        self.id = payload["raw_id"]
        self.label = payload.get("label")
        self.description = payload.get("description")
        self.risk_level = payload.get("risk_level")
        self.doc_title = payload.get("doc_title")
        self.doc_stage = payload.get("doc_stage")
        self.doc_mode = payload.get("doc_mode")
        self.item_stage = payload.get("item_stage", self.doc_stage)
        self.item_mode = payload.get("item_mode", self.doc_mode)
        self.source_type = "policy"
        self.source_path = payload.get("source_path")
        self.source_chunk_md = fix_raw_excerpt_md(
            payload.get("source_chunk_md"),
            self.source_path,
        )
        self.embedding = None  # 不再存本地 embedding


class ThesisSegmentFromPayload:
    def __init__(self, payload: dict):
        self.id = payload["raw_id"]
        self.label = payload.get("label", "")
        self.summary = payload.get("summary") or payload.get("description") or ""
        self.stage = payload.get("stage", payload.get("item_stage", "other"))
        self.mode = payload.get("mode", payload.get("item_mode", "precedents"))
        self.field = payload.get("field", "unknown")
        self.source_path = payload.get("source_path")
        self.doc_title = payload.get("doc_title")
        self.source_type = payload.get("source_type", "thesis")
        self.role = payload.get("role", "technical_precedent")
        self.domain_tags = payload.get("domain_tags", [])
        self.construct_tags = payload.get("construct_tags", [])
        self.user_tags = payload.get("user_tags", [])
        self.metric_tags = payload.get("metric_tags", [])
        raw_md = payload.get("raw_excerpt_md") or payload.get("source_chunk_md")
        self.source_chunk_md = fix_raw_excerpt_md(raw_md, self.source_path)
        self.embedding = None


class PolicyRepository:
    def __init__(self, client: QdrantClient, collection_name: str = "policy_docs"):
        self.client = client
        self.collection_name = collection_name

    def scored_search(
        self,
        query_emb: List[float],
        stage: str,
        mode: str,
        gap: str,
        top_k: int = 12,
    ) -> List[tuple[PolicyItemFromPayload, float]]:
        if not query_emb:
            return []

        # ① 调用 query_points，得到一个响应对象
        resp = self.client.query_points(
            collection_name=self.collection_name,
            query=query_emb,
            limit=60,
            with_payload=True,
        )
        # ② 真正的 hits 在 resp.points 里
        hits = resp.points
        print(f"{resp.points}")
        results = []
        print(f"hits:{hits}")
        for h in hits:
            p = h.payload or {}
            base_sim = h.score or 0.0
            bonus = 0.0
            item_stage = p.get("item_stage") or p.get("doc_stage")
            item_mode = p.get("item_mode") or p.get("doc_mode")

            if item_stage == stage:
                bonus += 0.10
            elif p.get("doc_stage") == stage:
                bonus += 0.05
            if item_mode == mode:
                bonus += 0.10
            elif p.get("doc_mode") == mode:
                bonus += 0.05

            if gap in ("process", "content") and mode in ("checklist", "diagnose", "ethics"):
                bonus += 0.05

            final_score = float(base_sim + bonus)
            item = PolicyItemFromPayload(p)
            results.append((item, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class ThesisRepository:
    def __init__(self, client: QdrantClient, collection_name: str = "thesis_segments"):
        self.client = client
        self.collection_name = collection_name

    def scored_search(
        self,
        query_emb: List[float],
        stage: str,
        mode: str,
        gap: str,
        top_k: int = 16,
    ) -> List[tuple[ThesisSegmentFromPayload, float]]:
        if not query_emb:
            return []

        resp = self.client.query_points(
            collection_name=self.collection_name,
            query=query_emb,
            limit=80,
            with_payload=True,
        )
        hits = resp.points

        results = []
        for h in hits:
            p = h.payload or {}
            base_sim = h.score or 0.0

            seg_stage = p.get("stage", "other")
            seg_mode = p.get("mode", "precedents")
            role = p.get("role", "technical_precedent")

            bonus = 0.0
            if seg_stage == stage:
                bonus += 0.08
            if seg_mode == mode:
                bonus += 0.08
            if mode in ("precedents", "exploration"):
                bonus += 0.12
            if gap == "precedent":
                bonus += 0.10
            if role == "technical_precedent":
                bonus += 0.02

            final_score = float(base_sim + bonus)
            item = ThesisSegmentFromPayload(p)
            results.append((item, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# -----------------------
# EvidenceCard + 融合
# -----------------------

class EvidenceCard:
    def __init__(
            self,
            evid_id: str,
            title: str,
            snippet: str,
            source_type: Literal["policy", "thesis"],
            meta: Dict[str, Any],
    ):
        self.id = evid_id
        self.title = title
        self.snippet = snippet
        self.source_type = source_type
        self.meta = meta


def llm_rerank_evidence(
        query_text: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 16,
) -> Dict[str, Dict[str, Any]]:
    """
    Self-RAG 风格的 rerank：
    - 对每个 candidate 给出 helpfulness ∈ [0,1]
    - 同时判断 role: rubric / precedent / other
    返回:
    { evid_id: {"score": float, "role": str, "tags": [...]} }
    """
    if not candidates:
        return {}

    def trunc(s: str, max_len: int = 420) -> str:
        s = s.strip()
        return s[: max_len - 3] + "..." if len(s) > max_len else s

    cand_briefs = [
        {
            "id": c["id"],
            "title": c.get("title", ""),
            "source_type": c.get("source_type", ""),
            "snippet": trunc(c.get("snippet", "")),
        }
        for c in candidates[: top_k * 2]
    ]

    system_prompt = (
        "You are a passage selector for EBCS—an Evidence-Bound, Rubric-Aligned "
        "Conversational Scaffolding system for MSc IDE graduation projects at TU Delft.\n"
        "EBCS uses TWO repositories:\n"
        "- 'policy': official IDE graduation rubrics, handbooks, stage checklists, templates.\n"
        "- 'thesis': segments from past IDE MSc theses used as precedents.\n\n"
        "Given the student's query and several snippets, estimate how helpful each snippet is "
        "for giving *actionable supervision* that is aligned with IDE rubrics.\n\n"
        "Interpretation of roles:\n"
        "- 'rubric': assessment criteria, checklists, required elements, templates, risk rules.\n"
        "- 'precedent': concrete thesis examples (methods, metrics, research questions, pitfalls).\n"
        "- 'other': background, meta advice, or content that is not directly usable for actions.\n\n"
        "For each candidate, output:\n"
        "- 'id': same as input id;\n"
        "- 'helpfulness': float in [0,1] (1 = extremely helpful and specific, 0 = useless/off-topic);\n"
        "- 'role': one of ['rubric','precedent','other'];\n"
        "- 'gap_tags': a small list chosen from ['content','process','knowledge','precedent'].\n"
        "  · 'content' = helps improve the draft text or RQ itself;\n"
        "  · 'process' = helps with stage checklists, planning, or workflow;\n"
        "  · 'knowledge' = provides conceptual/technical background the student likely lacks;\n"
        "  · 'precedent' = gives concrete IDE thesis examples or method↔metric patterns.\n\n"
        "Return ONLY a JSON array, like:\n"
        "[\n"
        "  {\"id\": \"policy:12\", \"helpfulness\": 0.93, \"role\": \"rubric\", "
        "\"gap_tags\": [\"process\",\"content\"]},\n"
        "  {\"id\": \"thesis:7\", \"helpfulness\": 0.81, \"role\": \"precedent\", "
        "\"gap_tags\": [\"precedent\"]}\n"
        "]"
    )

    user_payload = {
        "student_query": query_text,
        "candidates": cand_briefs,
    }
    user_prompt = json.dumps(user_payload, ensure_ascii=False)

    try:
        resp = call_llm(system_prompt, user_prompt)
        txt = resp.strip()
        data = json.loads(txt)
    except Exception:
        return {}

    results: Dict[str, Dict[str, Any]] = {}
    for obj in data:
        cid = obj.get("id")
        if not cid:
            continue
        try:
            h = float(obj.get("helpfulness", 0.0))
        except Exception:
            h = 0.0
        h = max(0.0, min(1.0, h))
        role = obj.get("role", "other")
        tags = obj.get("gap_tags", [])
        if isinstance(tags, str):
            tags = [tags]
        results[cid] = {
            "score": h,
            "role": role,
            "tags": tags,
        }
    return results


from functools import lru_cache


def log_chat_turn(
        user_id: str,
        round_index: int,
        timestamp_question: str,
        timestamp_answer: str,
        question: str,
        plan_obj: "CoachPlan",
        evidence_cards: List[EvidenceCard],
        alignment: Dict[str, Any],
):
    """
    仿 baseline 的 chat_turns_baseline.csv：
    一行 = 一轮 coach plan 的问答（question + plan）。
    """
    try:
        row = {
            "user_id": user_id,
            "turn_index": round_index,
            "timestamp_question": timestamp_question,
            "timestamp_answer": timestamp_answer,
            "question": question,
            # 为了对齐 baseline，这里放 overview + JSON 两个字段
            "answer_overview": plan_obj.overview,
            "answer_json": plan_obj.model_dump_json(),
            # 当前 routing 信息
            "stage": alignment.get("stage"),
            "mode": alignment.get("mode"),
            "gap": alignment.get("gap"),
            # 本轮所有 evidence 的扁平信息
            "retrieved_ids": ";".join(e.id for e in evidence_cards),
            "retrieved_source_types": ";".join(e.source_type for e in evidence_cards),
            "retrieved_doc_titles": ";".join(
                (e.meta.get("doc_title") or "").replace(";", ",") for e in evidence_cards
            ),
            "retrieved_scores": ";".join(
                f"{e.meta.get('score', 0):.4f}" for e in evidence_cards
            ),
        }
        append_csv_row(
            CHAT_LOG_FILE,
            fieldnames=list(row.keys()),
            row_dict=row,
        )
    except Exception as e:
        print("Failed to log EBCS chat turn:", e)


def log_evidence_event(
        user_id: str,
        round_index: int,
        event_type: str,
        evid: EvidenceCard | None = None,
        rec_index: int | None = None,
        rec_title: str | None = None,
        extra: Dict[str, Any] | None = None,
):
    """
    仿 baseline 的 snippet_clicks_baseline.csv：
    - event_type: 'evidence_click', 'recommendation_expand', 'source_section_expand' 等
    - evid: 对应的 EvidenceCard（如果有）
    - rec_index / rec_title: 属于第几个 recommendation
    """
    try:
        extra = extra or {}
        row = {
            "timestamp_click": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "turn_index": round_index,  # 用相同列名对齐 baseline
            "event_type": event_type,  # show/hide/click/expand…
            # evidence 基本信息（如果有）
            "evid_id": getattr(evid, "id", None),
            "source_type": getattr(evid, "source_type", None),
            "doc_title": (evid.meta.get("doc_title") if evid else None),
            "score": (evid.meta.get("score") if evid else None),
            "helpful": (evid.meta.get("helpful") if evid else None),
            "tags": (evid.meta.get("tags") if evid else None),
            # 属于哪条 recommendation
            "rec_index": rec_index,
            "rec_title": rec_title,
            # 额外字段 JSON 化（以防你之后想加更多信息）
            "extra": json.dumps(extra, ensure_ascii=False),
        }
        append_csv_row(
            EVIDENCE_LOG_FILE,
            fieldnames=list(row.keys()),
            row_dict=row,
        )
    except Exception as e:
        print("Failed to log evidence event:", e)


import re


def _norm_heading(text: str) -> str:
    """Normalize heading text for matching (lowercase, collapse spaces, drop #)."""
    # remove leading # and surrounding spaces
    t = re.sub(r"^#+\s*", "", text).strip().lower()
    # collapse multiple spaces
    t = re.sub(r"\s+", " ", t)
    return t


def fuse_evidence(
        query_text: str,
        query_emb: List[float],
        stage: str,
        mode: str,
        gap: str,
        policy_repo: PolicyRepository,
        thesis_repo: ThesisRepository,
        total_k: int = 12,
) -> List[EvidenceCard]:
    """
    RAG-Fusion + Self-RAG 风格的 evidence fusion：
    1) 生成多条子查询（policy / precedent / mixed）
    2) 对每条子查询在两个仓库检索，使用 RRF + 子查询权重做 rank fusion
    3) 用 LLM reranker 估计 helpfulness + role/gap_tags
    4) 做带配额的 MMR set selection，保证 rubrics & precedents 兼有且多样
    """
    log_step("Step 2: start fuse_evidence (RAG retrieval)...")
    # ---------- 1) Multi-query 生成 ----------
    subqueries = generate_subqueries(
        task_context=query_text,
        stage=stage,
        mode=mode,
        gap=gap,
        max_queries=6,
    )

    # 没拿到就 fallback 单 query
    if not subqueries:
        subqueries = [
            {"id": "Q1", "text": query_text, "type": "mixed", "weight": 0.8},
        ]

    # ---------- 2) multi-query × dual-repo 检索 + RRF ----------
    def rrf(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank + 1)

    fused: Dict[str, Dict[str, Any]] = {}

    def add_candidate(evid_key: str, source_type: str, item, base_score: float, rrf_score: float):
        if evid_key not in fused or base_score > fused[evid_key]["base_score"]:
            fused[evid_key] = {
                "id": evid_key,
                "item": item,
                "source_type": source_type,
                "base_score": float(base_score),
                "rrf_score": float(rrf_score),
            }
        else:
            # 同一 doc 被不同子查询命中时，累加 rrf
            fused[evid_key]["rrf_score"] += float(rrf_score)

    for q in subqueries:
        q_text = q["text"]
        q_type = q["type"]
        w_q = q["weight"]

        emb = embed_text(q_text)

        # 根据 q_type 调整两个 repo 的权重
        if q_type == "policy":
            w_policy, w_thesis = 1.0, 0.5
        elif q_type == "precedent":
            w_policy, w_thesis = 0.6, 1.0
        else:  # mixed
            w_policy = w_thesis = 0.9

        # 每个子查询里多取一点，再交给后面的融合
        p_scored = policy_repo.scored_search(emb, stage, mode, gap, top_k=10)
        t_scored = thesis_repo.scored_search(emb, stage, mode, gap, top_k=12)

        for rank, (it, s) in enumerate(p_scored):
            evid_key = f"policy:{it.id}"
            score_rrf = w_q * w_policy * rrf(rank)
            add_candidate(evid_key, "policy", it, base_score=float(s), rrf_score=score_rrf)

        for rank, (seg, s) in enumerate(t_scored):
            evid_key = f"thesis:{seg.id}"
            score_rrf = w_q * w_thesis * rrf(rank)
            add_candidate(evid_key, "thesis", seg, base_score=float(s), rrf_score=score_rrf)

    if not fused:
        return []

    # 把 base_score 和 rrf_score 合成一个初始分数
    for d in fused.values():
        # 这里简单线性组合：stage/mode/gap bonus 已经在 base_score 里
        d["score"] = float(0.6 * d["base_score"] + 1.2 * d["rrf_score"])

    # ---------- 3) LLM rerank（Self-RAG 风格 helpfulness） ----------
    cand_list_for_llm = []
    for key, d in fused.items():
        it = d["item"]
        if d["source_type"] == "policy":
            title = getattr(it, "label", "")
            snippet = getattr(it, "description", "")
        else:
            title = getattr(it, "label", "")
            snippet = getattr(it, "summary", "")
        cand_list_for_llm.append(
            {
                "id": key,
                "title": title,
                "snippet": snippet,
                "source_type": d["source_type"],
            }
        )

    llm_info = llm_rerank_evidence(query_text, cand_list_for_llm, top_k=total_k * 2)

    # 把 helpfulness/role/gap_tags 注入 fused
    for key, d in fused.items():
        info = llm_info.get(key)
        if info:
            helpful = info.get("score", 0.0)
            role = info.get("role", None)
            tags = info.get("tags", [])
            d["llm_helpful"] = float(helpful)
            d["llm_role"] = role
            d["llm_tags"] = tags
            # final score 乘一个因子（Self-RAG idea）
            d["score"] = float(d["score"] * (0.5 + 0.8 * helpful))
        else:
            d["llm_helpful"] = 0.0
            d["llm_role"] = None
            d["llm_tags"] = []

    # ---------- 4) 带配额的 MMR set selection ----------
    # 至少想要多少 rubrics / precedents
    min_policy = 2
    min_thesis = 3

    # 按最终 score 排序
    candidates_sorted = sorted(fused.values(), key=lambda x: x["score"], reverse=True)

    def get_embedding(d: Dict[str, Any]) -> np.ndarray:
        it = d["item"]
        return getattr(it, "embedding", None)

    selected: List[Dict[str, Any]] = []
    lambda_div = 0.4

    for cand in candidates_sorted:
        if len(selected) >= total_k:
            break
        emb_c = get_embedding(cand)
        if emb_c is None or not selected:
            selected.append(cand)
            continue

        sims = []
        for s in selected:
            emb_s = get_embedding(s)
            if emb_s is None:
                continue
            denom = (np.linalg.norm(emb_c) * np.linalg.norm(emb_s) + 1e-9)
            sims.append(float(np.dot(emb_c, emb_s) / denom))
        max_sim = max(sims) if sims else 0.0

        mmr_score = cand["score"] - lambda_div * max_sim
        if mmr_score > 0.0 or max_sim < 0.85:
            selected.append(cand)

    # 如果配额没满足，尝试从剩余中补齐
    def count_by_type(lst, t):
        return sum(1 for d in lst if d["source_type"] == t)

    # 补 policy
    if count_by_type(selected, "policy") < min_policy:
        for cand in candidates_sorted:
            if cand in selected:
                continue
            if cand["source_type"] != "policy":
                continue
            selected.append(cand)
            if count_by_type(selected, "policy") >= min_policy:
                break

    # 补 thesis
    if count_by_type(selected, "thesis") < min_thesis:
        for cand in candidates_sorted:
            if cand in selected:
                continue
            if cand["source_type"] != "thesis":
                continue
            selected.append(cand)
            if count_by_type(selected, "thesis") >= min_thesis:
                break

    # 最终截断并按 score 排序
    selected = sorted(selected, key=lambda x: x["score"], reverse=True)[:total_k]

    # ---------- 5) 构造 EvidenceCard ----------
    cards: List[EvidenceCard] = []
    for idx, d in enumerate(selected, 1):
        it = d["item"]
        src_type = d["source_type"]
        score = d["score"]
        llm_help = d.get("llm_helpful", 0.0)
        llm_role = d.get("llm_role")
        llm_tags = d.get("llm_tags", [])

        evid_id = f"P{idx}" if src_type == "policy" else f"T{idx}"

        if src_type == "policy":
            title = f"{it.label} (rubric)"
            snippet = it.description
            meta = {
                "raw_id": it.id,
                "doc_title": it.doc_title,
                "stage": it.doc_stage,
                "mode": it.doc_mode,
                "risk": it.risk_level,
                "score": round(score, 3),
                "helpful": round(llm_help, 2),
                "llm_role": llm_role,
                "tags": ",".join(llm_tags),
                "source_path": getattr(it, "source_path", None),
                "source_chunk_md": getattr(it, "source_chunk_md", None),
            }
        else:
            title = f"{it.label} (precedent)"
            snippet = it.summary
            print("DEBUG in fuse_evidence:", it.id, getattr(it, "source_chunk_md", None))
            meta = {
                "raw_id": it.id,
                "doc_title": it.doc_title,
                "field": it.field,
                "stage": it.stage,
                "mode": it.mode,
                "score": round(score, 3),
                "helpful": round(llm_help, 2),
                "llm_role": llm_role,
                "tags": ",".join(llm_tags),
                "source_path": getattr(it, "source_path", None),
                "source_chunk_md": getattr(it, "source_chunk_md", None),
            }

        cards.append(
            EvidenceCard(
                evid_id=evid_id,
                title=title,
                snippet=snippet,
                source_type=src_type,
                meta=meta,
            )
        )
    log_step(f"Step 2 done: fused {len(cards)} evidence items.")
    return cards


# -----------------------
# 路由 + Gap 追问
# -----------------------

def route_and_maybe_ask(
        conversation_text: str,
) -> Dict[str, Any]:
    """
    Router：推断 stage / mode / gap，并判断是否信息足够。
    专门针对 TU Delft IDE MSc graduation（rubrics + theses 双语料）。
    """
    stage_list = ", ".join(STAGES)
    mode_list = ", ".join(MODES)
    gap_list = ", ".join(GAPS)

    system_prompt = (
        "You are the intake router for EBCS, an evidence-bound thesis coach for MSc IDE "
        "graduation projects at TU Delft.\n"
        "The system uses:\n"
        "- 'policy_docs': official TU Delft IDE MSc graduation rubrics, handbooks, checklists;\n"
        "- 'thesis_corpus': past IDE MSc graduation theses as precedents.\n\n"
        "You NEVER need to ask about degree level, language, or country: it is always an "
        "IDE MSc graduation project at TU Delft.\n\n"
        "Your job:\n"
        f"1) Infer the student's current thesis stage from: {stage_list}.\n"
        "   · 'proposal': exploring topic / early problem framing;\n"
        "   · 'greenlight': preparing or revising a formal research plan for approval;\n"
        "   · 'midterm': in the middle of thesis work, methods partly running, results emerging;\n"
        "   · 'final': writing up / polishing the final report or design dossier;\n"
        "   · 'defense': preparing for the final defense / presentation;\n"
        "   · 'other': anything that clearly does not fit the above.\n\n"
        f"2) Infer the dominant conversation mode from: {mode_list}.\n"
        "   · 'exploration': clarify topic, problem space, or directions;\n"
        "   · 'precedents': look for similar theses, methods, or cases;\n"
        "   · 'diagnose': debug what is wrong / stuck in the thesis (RQ, method, results);\n"
        "   · 'checklist': work through stage-specific requirements and rubrics;\n"
        "   · 'plan_synthesis': synthesize a concrete next-steps plan or research design;\n"
        "   · 'critique': critique a draft, concept, method, or RQ;\n"
        "   · 'ethics': discuss ethics, privacy, data protection, or risk escalation;\n"
        "   · 'defense_drill': rehearse exam/defense questions;\n"
        "   · 'other': generic chat or not clear.\n\n"
        f"3) Infer the primary gap from: {gap_list}.\n"
        "   · 'content': the student has some draft text / slides / plan that needs improvement;\n"
        "   · 'process': the student is unsure about procedures, requirements, or next steps;\n"
        "   · 'knowledge': the student mainly lacks conceptual or methodological background;\n"
        "   · 'precedent': the student mainly wants examples from previous theses;\n"
        "   · 'mixed': multiple gaps are equally strong;\n"
        "   · 'unknown': impossible to tell from the message.\n\n"
        "4) Decide if there is ENOUGH information to retrieve rubrics + precedents.\n"
        "   There is enough info if you know AT LEAST:\n"
        "   · the project domain / context (what kind of system/solution or topic),\n"
        "   · the main users or stakeholders,\n"
        "   · and 1–2 key evaluation dimensions or metrics (e.g., usability, task success, adoption).\n"
        "   For 'content' gap you also need to know whether some draft already exists.\n"
        "   For 'precedent' gap you also need to know at least a rough method or topic focus.\n\n"
        "If there is NOT enough information, ask EXACTLY ONE very concrete follow-up question.\n"
        "The follow-up should be:\n"
        "  - short (1–2 sentences),\n"
        "  - focused on the MOST important missing pieces,\n"
        "  - answerable in 1–3 sentences.\n\n"
        "Also label what is missing, using a list drawn from:\n"
        "['domain', 'users', 'metrics', 'draft_status', 'method', 'other'].\n\n"
        "Return ONLY a JSON object with this schema:\n"
        "{\n"
        "  \"stage\": \"proposal|greenlight|midterm|final|defense|other\",\n"
        "  \"mode\": \"exploration|precedents|diagnose|checklist|plan_synthesis|critique|ethics|defense_drill|other\",\n"
        "  \"gap\": \"content|process|knowledge|precedent|mixed|unknown\",\n"
        "  \"enough_info\": true/false,\n"
        "  \"missing\": [\"domain\", \"users\"],\n"
        "  \"reason\": \"short explanation of your routing (≤3 clauses)\",\n"
        "  \"followup_question\": \"one concrete follow-up question in Chinese\"\n"
        "}\n"
        "Do NOT include any markdown. Output pure JSON."
    )
    user_prompt = conversation_text

    resp = call_llm(system_prompt, user_prompt)
    txt = resp.strip()

    try:
        data = json.loads(txt)
        return {
            "stage": data.get("stage", "proposal"),
            "mode": data.get("mode", "exploration"),
            "gap": data.get("gap", "unknown"),
            "enough_info": bool(data.get("enough_info", False)),
            "missing": data.get("missing", []),
            "reason": data.get("reason", ""),
            "followup_question": data.get("followup_question", "").strip(),
        }
    except Exception:
        # 智能 fallback：让 LLM 根据用户输入总结最缺的槽位并生成追问
        smart_fallback_prompt = (
            "You are the fallback router for EBCS. The main router returned invalid JSON.\n"
            "Now YOU must infer:\n"
            "-"
            "From the following student's message, extract what is still unclear among:\n"
            "  domain / users / metrics / draft_status / method.\n"
            "- Then write ONE follow-up question in Chinese.\n"
            "- The question must be specific, answerable in 1–3 sentences.\n"
            "- It must directly target the MOST critical missing info.\n"
            "- Do NOT output JSON. Only output the question text.\n"
        )
        try:
            fq = call_llm(
                smart_fallback_prompt,
                conversation_text,
            ).strip()
        except Exception:
            # 最后兜底：如果 LLM 又挂了
            fq = (
                "为了继续，我需要再了解一点：你的项目主要面向谁、在什么场景使用？"
                "另外，你现在最想验证的 1–2 个指标是什么？"
            )

        return {
            "stage": "proposal",
            "mode": "exploration",
            "gap": "unknown",
            "enough_info": False,
            "missing": ["domain", "users", "metrics"],
            "reason": "Fallback router engaged due to JSON parse failure.",
            "followup_question": fq,
        }


def generate_subqueries(
        task_context: str,
        stage: str,
        mode: str,
        gap: str,
        max_queries: int = 6,
) -> List[Dict[str, Any]]:
    """
    RAG-Fusion 风格的子查询生成：
    针对 TU Delft IDE MSc：policy_docs + thesis_corpus。
    """
    system_prompt = (
        "You are the sub-query planner for EBCS, a dual-repository RAG system for IDE MSc "
        "graduation projects at TU Delft.\n"
        "Repositories:\n"
        "- 'policy': official IDE rubrics, graduation handbook pages, stage checklists, templates.\n"
        "- 'thesis': segments from past IDE MSc theses (methods, metrics, RQs, pitfalls, examples).\n\n"
        "Given the student's notes plus the routing (stage/mode/gap), propose 3–6 short English "
        "sub-queries for retrieval.\n"
        "Each sub-query must target one of: 'policy', 'precedent', or 'mixed'.\n\n"
        "Guidance:\n"
        "- Use 'policy' for queries about requirements, assessment criteria, checklists, deadlines,\n"
        "  mandatory elements of proposal/green-light/midterm/final/defense.\n"
        "- Use 'precedent' for queries about similar IDE theses, methods, metrics, research questions,\n"
        "  data collection, analysis pipelines, or typical pitfalls.\n"
        "- Use 'mixed' when both rubrics and precedents are equally important.\n"
        "- Tailor queries to the given stage and mode; e.g., at 'proposal/exploration', focus on\n"
        "  framing RQs and domain exemplars; at 'greenlight/checklist', focus on required slots\n"
        "  (population, context, method↔metric, feasibility, ethics).\n\n"
        "Return ONLY a JSON object like:\n"
        "{\n"
        "  \"queries\": [\n"
        "     {\"id\": \"Q1\", \"text\": \"IDE MSc proposal rubric for research plan and RQ quality\", "
        "\"type\": \"policy\", \"weight\": 0.9},\n"
        "     {\"id\": \"Q2\", \"text\": \"similar IDE theses about warehouse collaboration and UX metrics\", "
        "\"type\": \"precedent\", \"weight\": 0.85}\n"
        "  ]\n"
        "}\n\n"
        f"Constraints:\n"
        f"- total queries between 3 and {max_queries};\n"
        "- 'text' should be ≤ 25 words;\n"
        "- 'type' ∈ ['policy','precedent','mixed'];\n"
        "- 'weight' is a float in [0.3,1.0] representing importance (higher = more important).\n"
        "Do NOT include markdown. Output pure JSON."
    )

    user_payload = {
        "stage": stage,
        "mode": mode,
        "gap": gap,
        "student_notes": task_context[-3500:],  # 防止太长
    }
    user_prompt = json.dumps(user_payload, ensure_ascii=False)

    try:
        resp = call_llm(system_prompt, user_prompt)
        txt = resp.strip()
        data = json.loads(txt)
        queries = data.get("queries", [])
    except Exception:
        base_q = task_context.split("\n")[-1][:120]
        queries = [
            {
                "id": "Q1",
                "text": f"IDE MSc graduation assessment criteria for stage {stage} at TU Delft",
                "type": "policy",
                "weight": 0.9,
            },
            {
                "id": "Q2",
                "text": f"similar IDE MSc theses and methods about: {base_q}",
                "type": "precedent",
                "weight": 0.8,
            },
            {
                "id": "Q3",
                "text": f"process checklist and required deliverables for stage {stage}",
                "type": "policy",
                "weight": 0.7,
            },
        ]

    cleaned = []
    for i, q in enumerate(queries):
        text = str(q.get("text", "")).strip()
        if not text:
            continue
        q_type = q.get("type", "mixed")
        if q_type not in ("policy", "precedent", "mixed"):
            q_type = "mixed"
        w = q.get("weight", 0.7)
        try:
            w = float(w)
        except Exception:
            w = 0.7
        w = max(0.3, min(1.0, w))
        cleaned.append(
            {
                "id": q.get("id", f"Q{i + 1}"),
                "text": text,
                "type": q_type,
                "weight": w,
            }
        )

    if len(cleaned) < 3:
        base_q = task_context.split("\n")[-1][:120]
        while len(cleaned) < 3:
            cleaned.append(
                {
                    "id": f"Q{len(cleaned) + 1}",
                    "text": base_q,
                    "type": "mixed",
                    "weight": 0.6,
                }
            )
    return cleaned[:max_queries]


def build_followup_question(
        task_context: str,
        routing: Dict[str, Any],
        followup_count: int,
) -> str:
    """
    使用模型来生成澄清问题：
    输出：中文的一段/一句 follow-up 问题，不要 JSON。
    更明确地围绕 domain / users / metrics / draft / method。
    """
    base_q = (routing.get("followup_question") or "").strip()
    missing = routing.get("missing", []) or []
    reason = routing.get("reason", "")

    payload = {
        "student_notes": task_context[-4000:],
        "routing": {
            "stage": routing.get("stage"),
            "mode": routing.get("mode"),
            "gap": routing.get("gap"),
            "missing": missing,
            "reason": reason,
        },
        "router_suggested_question": base_q,
        "followup_count": followup_count,
    }

    system_prompt = (
        "You are the follow-up question designer for EBCS, an evidence-bound thesis coach for "
        "TU Delft IDE MSc graduation projects.\n"
        "Input is a JSON payload with:\n"
        "- student's notes (Chinese/English),\n"
        "- routing result (stage/mode/gap/missing slots),\n"
        "- router_suggested_question: a draft question from another agent (Chinese),\n"
        "- followup_count: how many clarifying turns have already been asked.\n\n"
        "Your goal is to output EXACTLY ONE short follow-up question in Chinese.\n"
        "Requirements:\n"
        "- 1–2 sentences max；\n"
        "- 紧扣 missing 列表中最关键的 1–3 个信息（例如 domain / users / metrics / draft_status / method）；\n"
        "- 让学生可以用 1–3 句话回答；\n"
        "- 如果 followup_count 为 0–1，可以更细一点；>=2 时，把多个关键点合并到一个问题里，减少来回；\n"
        "- 可以参考 router_suggested_question 的意图，但要重写得更自然、更具体。\n\n"
        "不要输出 JSON，不要解释你的思路，只输出最终的提问句子。"
    )
    user_prompt = json.dumps(payload, ensure_ascii=False)

    try:
        resp = call_llm(system_prompt, user_prompt)
        text = resp.strip()
        if not text:
            if base_q:
                return base_q
            return (
                "为了更好地用系里的 rubrics 帮你检索先例，"
                "可以请你用 2–4 句话补充一下：项目大概做什么、主要用户是谁，"
                "以及你最想优化或评估的 1–2 个指标（例如可用性、体验质量、采纳率等）吗？"
            )
        return text
    except Exception:
        if base_q:
            return base_q
        return (
            "为了给你更具体的建议，可以补充一下：项目大概做什么、主要用户是谁、"
            "以及你现在最关心的 1–2 个评价指标是什么？"
        )


def build_exhausted_warning(
        task_context: str,
        alignment: Dict[str, Any],
        followup_count: int,
) -> str:
    """
    Called when we already asked several clarification questions but the project
    is still underspecified.

    This function does NOT run any retrieval. It only produces a short
    explanatory message to the student, telling them that:
    - the information is still too high-level to fully align with the IDE rubrics,
    - the system will now proceed with reasonable assumptions based on typical
      IDE MSc cases,
    - and what kind of information would make future guidance more tailored.
    """

    payload = {
        "student_notes": task_context[-4000:],  # keep it compact
        "alignment": {
            "stage": alignment.get("stage"),
            "mode": alignment.get("mode"),
            "gap": alignment.get("gap"),
        },
        "followup_count": followup_count,
    }

    system_prompt = (
        "You are the meta-coach narrator for EBCS, an evidence-bound thesis coach for "
        "MSc IDE graduation projects at TU Delft.\n"
        "You are called when the system has already asked several clarification questions, "
        "but the project is still underspecified.\n\n"
        "Input is a JSON payload with:\n"
        "- student's notes (free text),\n"
        "- current alignment (stage/mode/gap),\n"
        "- followup_count (usually ≥ 3 at this point).\n\n"
        "Your job is to produce a SHORT message in English (2–4 short paragraphs) that:\n"
        "1) Gently explains that the information is still quite high-level and that several rubric "
        "   slots (e.g., problem domain, users, metrics, method) remain unclear, so the coach "
        "   cannot give a fully tailored, context-specific plan yet.\n"
        "2) Clearly states that from this point on, the system will proceed by making reasonable "
        "   assumptions based on typical IDE MSc graduation projects and the programme’s rubrics, "
        "   and will still try to provide a generic but actionable plan.\n"
        "   Make it explicit that these assumptions are based on common patterns in past IDE theses "
        "   and rubrics, not on a perfect match to this specific project.\n"
        "3) Paraphrases, in natural language, the currently inferred stage, mode, and main type of "
        "   difficulty (gap), so the student has an intuitive sense of where they roughly are "
        "   (e.g., early proposal, stuck on research question, unsure about methods, etc.).\n"
        "4) Ends with 2–3 bullet-like suggestions (you can format them as lines starting with '- ') "
        "   about what concrete information the student should prepare next time in order to receive "
        "   more precise, rubric-aligned coaching (e.g., clearer project context, key users, a draft "
        "   research question, intended methods, and 1–2 evaluation metrics).\n\n"
        "Important:\n"
        "- You only produce this explanatory message; retrieval over rubrics and theses will happen "
        "  in later steps of the pipeline.\n"
        "- Do NOT output JSON.\n"
        "- Do NOT describe your internal reasoning; just write the final message to the student in "
        "  clear, friendly English."
    )

    user_prompt = json.dumps(payload, ensure_ascii=False)

    try:
        resp = call_llm(system_prompt, user_prompt)
        text = resp.strip()
        if not text:
            raise ValueError("empty warn text")
        return text
    except Exception:
        # Intelligent fallback: regenerate a natural warning message via LLM
        fallback_prompt = (
            "You are generating a fallback warning for an IDE MSc thesis coaching system.\n"
            "The primary generator failed, so you must produce a short, warm, helpful "
            "message in Chinese.\n\n"
            "You MUST:\n"
            "1) Acknowledge that the system has already asked multiple follow-up questions, but key information is still unclear.\n"
            "2) Based on the student's input (below), infer which items are still missing (e.g., domain / users / metrics / method).\n"
            "3) Explain that you will now proceed using reasonable assumptions typical for IDE MSc theses.\n"
            "4) Restate the inferred stage / mode / gap in natural Chinese.\n"
            "5) End with 2–3 bullet-point suggestions telling the student what concrete info to provide next time.\n\n"
            "Output text only, no JSON.\n"
        )
        try:
            out = call_llm(
                fallback_prompt,
                json.dumps(payload, ensure_ascii=False),
            ).strip()
            if out:
                return out
        except:
            pass

        # Final fallback (rarely triggered)
        return (
            "The information is still too high-level for me to align accurately with IDE thesis rubrics.\n\n"
            "I will now make some reasonable assumptions based on common IDE MSc projects and give you a general but actionable plan.\n\n"
            "Next time, if you can provide clearer project context, main users, research focus, and intended methods/metrics, "
            "I can give you much more tailored guidance."
        )


# -----------------------
# Evidence-bound 回复：JSON 结构化输出
# -----------------------
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    统一封装 Responses API 调用：
    - system_prompt: 系统指令
    - user_prompt:   用户/上下文内容（可以很长）
    返回：模型文本输出（已经拼接好）
    """
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": system_prompt}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt}
                ],
            },
        ],
    )

    # 新 SDK 通常有 output_text，稳一点就两种都支持
    if hasattr(resp, "output_text"):
        return resp.output_text
    # fallback：从 output 里拼
    texts = []
    for o in resp.output:
        for c in o.content:
            if getattr(c, "type", "") == "output_text":
                texts.append(c.text)
    return "".join(texts)


from pydantic import BaseModel, Field
from typing import List, Optional

class Recommendation(BaseModel):
    title: str
    evidence_ids: List[str] = Field(default_factory=list)
    reason: str
    action: str


class CoachPlan(BaseModel):
    overview: str
    recommendations: List[Recommendation]
    follow_up: Optional[str] = None


def generate_coach_plan(
        user_input: str,
        stage: str,
        mode: str,
        gap: str,
        task_context: str,
        evidence_cards: List[EvidenceCard],
        history: List[Dict[str, str]],
) -> CoachPlan:
    """
    使用 responses.parse + Pydantic，直接拿到结构化的 CoachPlan 对象。
    """
    evid_text = "\n".join(
        f"[{e.id}] {e.title}\n{e.snippet}" for e in evidence_cards
    ) or "(no evidence found)"

    hist = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in history[-6:])

    system_msg = {
        "role": "system",
        "content": (
            "You are EBCS, an evidence-bound, rubric-aligned graduation project coach for "
            "MSc IDE students at TU Delft.\n"
            "You ALWAYS:\n"
            "  - respect official IDE graduation rubrics, handbooks, and stage checklists;\n"
            "  - ground your advice in the provided evidence snippets (rubrics + thesis precedents);\n"
            "  - bind each recommendation to explicit evidence IDs.\n\n"
            "You will be given:\n"
            "- recent conversation turns,\n"
            "- the student's overall notes,\n"
            "- the new message,\n"
            "- and retrieved evidence snippets with IDs (e.g., P1/T3).\n\n"
            "Routing for this turn:\n"
            f"- stage = {stage}\n"
            f"- mode  = {mode}\n"
            f"- gap   = {gap}\n\n"
            "Your job is to fill the CoachPlan schema provided by the caller "
            "(overview + recommendations + optional follow_up).\n\n"
            "Guidelines:\n"
            "1) OVERVIEW\n"
            "   - 2–4 sentences summarizing where the student is, what is stuck, and the main priority.\n"
            "   - Mention the stage in natural language (e.g., 'early proposal', 'green-light prep').\n\n"
            "2) RECOMMENDATIONS (2–5 items is typical)\n"
            "   - Each Recommendation should be focused on ONE coherent issue (e.g., clarify RQ, "
            "     align method↔metrics, structuring the proposal, preparing for defense questions).\n"
            "   - 'evidence_ids': list of snippet IDs (e.g., ['P1','P3','T2']) that support this advice.\n"
            "   - 'reason': explain, in 2–4 sentences, how the cited rubrics and precedents justify "
            "     this recommendation. Be explicit about rubric slots (e.g., population, context, "
            "     method, metrics, ethics).\n"
            "   - 'action': give concrete next steps the student can do in 30–90 minutes, using clear "
            "     imperative language (\"Write...\", \"List...\", \"Decide...\", \"Compare...\").\n"
            "   - Prefer fewer, deeper recommendations over many superficial tips.\n\n"
            "3) FOLLOW_UP (optional)\n"
            "   - If helpful, suggest one focused topic or artefact for the next conversation turn "
            "     (e.g., \"Next time, paste your current RQ and method sketch so we can stress-test it.\").\n\n"
            "Important:\n"
            "- Stay within the constraints of the evidence: do not contradict rubrics or precedents.\n"
            "- If evidence is thin, be transparent in the 'reason' field (e.g., 'rubric is generic, so we...').\n"
            "- Do NOT explain the JSON schema—just produce content that fits it."
        ),
    }

    user_ctx = (
        f"Conversation so far:\n{hist}\n\n"
        f"Student's overall description / notes:\n{task_context}\n\n"
        f"New message:\n{user_input}\n\n"
        f"Retrieved evidence snippets with IDs:\n{evid_text}"
    )

    user_msg = {
        "role": "user",
        "content": user_ctx,
    }

    response = client.responses.parse(
        model=OPENAI_MODEL,
        input=[system_msg, user_msg],
        text_format=CoachPlan,
    )

    plan: CoachPlan = response.output_parsed
    return plan


# -----------------------
# Streamlit 状态管理
# -----------------------

# ========= 状态初始化 =========
@st.cache_resource
def load_repositories():
    client_q = get_qdrant_client()
    return PolicyRepository(client_q, POLICY_COLLECTION), ThesisRepository(client_q, THESES_COLLECTION)


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "alignment" not in st.session_state:
        st.session_state.alignment = {
            "stage": "proposal",
            "mode": "exploration",
            "gap": "unknown",
            "enough_info": False,
        }
    if "task_context" not in st.session_state:
        st.session_state.task_context = ""
    if "evidence_cards" not in st.session_state:
        st.session_state.evidence_cards = []
    if "followup_count" not in st.session_state:
        st.session_state.followup_count = 0
    # 当前被选中的 evidence（用于高亮右侧卡片）
    if "selected_evidence" not in st.session_state:
        st.session_state.selected_evidence = None
    # 是否正在生成回答
    if "busy" not in st.session_state:
        st.session_state.busy = False
    if "show_evidence_panel" not in st.session_state:
        st.session_state.show_evidence_panel = False
    if "round_id" not in st.session_state:
        st.session_state.round_id = 0


# -----------------------
# UI 辅助：渲染 Coach plan 为卡片
# -----------------------


def fix_raw_excerpt_md(raw_excerpt_md: str, source_path: str) -> str:
    """
    1）把本地相对 markdown 图片链接：
        [alt](images/...xyz)
        ![alt](images/...xyz)
       改成 S3 上的绝对路径；

    2）把所有 markdown 链接/图片里的 URL 里的空格替换成 %20；

    3a）如果发现形如  [xxx](https://... .jpg/.png) 这种“指向图片的普通链接”，
        自动改成图片语法  ![xxx](https://... .jpg/.png)

    3b）如果发现  alt 里有很多换行的超长图片写法，
        统一收缩成  ![image](url)  避免 markdown 解析失败。
    """

    if not raw_excerpt_md:
        return raw_excerpt_md

    md_text = raw_excerpt_md

    # ---------- 1. 处理本地相对路径 images/... -> S3 绝对路径 ----------
    if source_path:
        import os
        dir_path = os.path.dirname(source_path)

        # Drop "repo..." 前缀
        m = re.match(r"^repo[^/]*/(.*)$", dir_path)
        if m:
            dir_path_no_repo = m.group(1)
        else:
            dir_path_no_repo = dir_path

        IMG_BASE_URL = "https://delft-public-img.s3.eu-west-1.amazonaws.com/"
        prefix = IMG_BASE_URL + dir_path_no_repo.strip("/") + "/"

        # 匹配 [..](images/...) 或 ![..](images/...)
        pattern_local_img = re.compile(r'(!?\[[^\]]*\])\((?:\.?/)?(images/[^)\s]+)\)')

        def _repl_local(m: re.Match) -> str:
            alt = m.group(1)
            rel = m.group(2)
            rel = rel.replace(" ", "%20")
            full_url = prefix + rel
            return f"{alt}({full_url})"

        md_text = pattern_local_img.sub(_repl_local, md_text)

    # ---------- 2. 所有 markdown URL 里的空格 -> %20 ----------
    pattern_any_link = re.compile(r'(\]\()([^)]+)\)')

    def _repl_space(m: re.Match) -> str:
        url = m.group(2)
        safe_url = url.replace(" ", "%20")
        return f"]({safe_url})"

    md_text = pattern_any_link.sub(_repl_space, md_text)

    # ---------- 3a. 指向图片的普通链接 -> 图片语法 ----------
    pattern_link_to_img = re.compile(
        r'\[(?P<alt>[^\]]*)\]\((?P<url>https?://[^)\s]+\.(?:png|jpe?g|gif|svg))\)'
    )

    def _repl_link_to_img(m: re.Match) -> str:
        alt = m.group("alt").strip()
        url = m.group("url").strip()
        return f"![{alt}]({url})"

    md_text = pattern_link_to_img.sub(_repl_link_to_img, md_text)

    # ---------- 3b. 多行 alt 的图片，统一压缩成 ![image](url) ----------
    # 匹配：  ![ 任意多行文字 ](https://...jpg/png/gif/svg)
    pattern_multiline_img = re.compile(
        r'!\[(?P<alt>.*?)\]\((?P<url>https?://[^)\s]+\.(?:png|jpe?g|gif|svg))\)',
        re.DOTALL,
    )

    def _repl_multiline_img(m: re.Match) -> str:
        url = m.group("url").strip()
        # alt 直接用一个简单的占位，避免换行
        return f"![image]({url})"

    md_text = pattern_multiline_img.sub(_repl_multiline_img, md_text)

    return md_text


def toggle_evidence_panel(evid_id: str):
    """
    点击任意 evidence 按钮的统一逻辑：
    - 如果当前已经显示 & 选中的就是它 → 关闭 panel（相当于第二次点击）
    - 否则 → 打开 panel并选中该 evidence
    """
    prev = st.session_state.get("selected_evidence")
    showing = st.session_state.get("show_evidence_panel", False)
    if showing and prev == evid_id:
        # 第二次点同一个 → 关掉
        st.session_state.show_evidence_panel = False
        st.session_state.selected_evidence = None

    else:
        # 打开 + 切换选中对象
        st.session_state.show_evidence_panel = True
        st.session_state.selected_evidence = evid_id


def render_plan_as_cards(
        plan: Dict[str, Any],
        evidence_index: Dict[str, EvidenceCard],
        key_prefix: str = "",
):
    """JSON plan -> Action 卡片；Evidence 变成 title 下面的小按钮，点击后高亮右侧卡片。"""
    overview = plan.get("overview", "")
    if overview:
        st.markdown(f"**Coach：** {overview}")

    # 全局样式：按来源区分颜色
    st.markdown("""
    <style>
    /* thesis 证据：蓝色 */
    div[data-testid="stButton"][id*="-thesis"] > button {
        color: #60a5fa !important;
    }
    /* rubric 证据：橘色 */
    div[data-testid="stButton"][id*="-rubric"] > button {
        color: #f97316 !important;
    }
    .ebcs-card {
    }
    .ebcs-card-title-row {
        display: flex;
        align-items: baseline;
        gap: 8px;
        margin-bottom: 4px;
    }
    .ebcs-card-index {
        font-weight: 700;
        font-size: 18px;
        color: #f97316;
    }
    .ebcs-card-title {
        font-weight: 600;
        font-size: 16px;
    }
    .ebcs-evidence-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 4px 0 10px 26px; /* 稍微缩进一点，看起来像挂在 title 下 */
    }
    .ebcs-section-reason{
    font-size: 0.9rem;
    line-height: 1.5;
    margin-bottom:0.5rem;
    color: gray;
    }
        .ebcs-section-answer{
    font-size: 1rem;
    line-height: 1.5;
    margin-bottom:0.5rem
    }
    .ebcs-section-reason > b {
   color:gray
    }
        .ebcs-section-answer > b {
   color:gray
    }
    .ebcs-card-answers{
        margin-top: 1rem;
    }

/* 选中「包含 evchip 元素的父级 div」 */
div:has(> div[class*="st-key-evchip-"]) {
    display: flex !important;
    flex-direction: row !important;
    margin-top: 1rem !important;
}
div[data-testid="stVerticalBlock"]:has(> div > div > div > .ebcs-evidence-row) {
    gap: 0 !important;
}
div[class*="st-key-evchip-"] div[data-testid="stMarkdownContainer"] {
    font-size: 0.75rem !important;
}

/* 小按钮：行内显示、挤在一起 */
div.stButton > button {
    border-radius: 1rem;
    padding: 0.1rem 0.4rem;
}
div[class*="st-key-evchip-thesis"] button {
    background-color: #EFE8E2;
    color: #524941;
    border-color:#EFE8E2;
}
div[class*="st-key-evchip-rubric"] button {
    background-color: #DDE5E1;
    color:#3A4743;
    border-color:#DDE5E1;
}
    </style>
    """, unsafe_allow_html=True)

    recs = plan.get("recommendations", [])
    for idx, rec in enumerate(recs, 1):
        title = rec.get("title", f"Recommendation {idx}")
        eid_list = rec.get("evidence_ids", []) or []
        reason = rec.get("reason", "")
        action = rec.get("action", "")
        with st.expander(f" Step {idx}. {title}", expanded=False):
            # ✅ 第一次展开时记录事件
            user_id = st.session_state.get("user_id")
            round_idx = st.session_state.get("round_id", 0)
            expand_flag_key = f"logged_rec_expand_{round_idx}_{idx}"

            if user_id and not st.session_state.get(expand_flag_key, False):
                log_evidence_event(
                    user_id=user_id,
                    round_index=round_idx,
                    event_type="recommendation_expand",
                    evid=None,  # 这里是整条 recommendation，不对应单一 evidence
                    rec_index=idx,
                    rec_title=title,
                )
                st.session_state[expand_flag_key] = True
            # with st.container():
            # 卡片外框 + 标题
            st.markdown(
                f"""
<div class="ebcs-card">
  <div class="ebcs-card-title-row">
    <div class="ebcs-card-index">{idx}.</div>
    <div class="ebcs-card-title">{title}</div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )
            # ---- Title 正下方的 evidence 按钮行 ----
            if eid_list:
                if idx == 1:
                    st.markdown(
                        "<p style='color: #f8bfbf;padding-top: 0.8rem;margin-bottom: 0.5rem;'>Click below the Evidence button to show the raw text; double-click to hide it.</p>",
                        unsafe_allow_html=True)
                # 开一个 row 容器，主要用于 margin 控制
                st.markdown('<div class="ebcs-evidence-row">', unsafe_allow_html=True)
                # 用一个空的 container 承载多个 st.button
                btn_row = st.container()
                with btn_row:
                    for eid in eid_list:
                        ev = evidence_index.get(eid)
                        if ev is None:
                            continue

                        is_selected = (
                                st.session_state.get("selected_evidence") == eid
                        )
                        src = "rubric" if ev.source_type == "policy" else "thesis"
                        label = f"{eid} · {src}"

                        btn_type = "primary" if is_selected else "secondary"
                        key = f"evchip-{src}-{key_prefix}-{idx}-{eid}"

                        # 单个小按钮
                        clicked = st.button(label, key=key, type=btn_type)
                        if clicked:
                            toggle_evidence_panel(eid)

                            # ✅ 记录 evidence click 事件
                            user_id = st.session_state.get("user_id")
                            round_idx = st.session_state.get("round_id", 0)
                            log_evidence_event(
                                user_id=user_id,
                                round_index=round_idx,
                                event_type="evidence_click",
                                evid=ev,
                                rec_index=idx,
                                rec_title=title,
                                extra={"button_label": label},
                            )

                            st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

            # ---- Reason / Action 放在同一张卡里，视觉上在按钮下面 ----
            st.markdown(
                f"""
<div class="ebcs-card-answers">
  <div class="ebcs-section-reason">
    <b>Reason</b>: {reason}
  </div>
  <div class="ebcs-section-answer">
    <b>Action</b>: {action}
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )

    follow_up = plan.get("follow_up", "")
    if follow_up:
        st.markdown(f"_Suggested next steps:{follow_up}_")


def build_export_text(plan: Dict[str, Any]) -> str:
    lines = []
    if plan.get("overview"):
        lines.append("Overview:")
        lines.append(plan["overview"])
        lines.append("")
    lines.append("Recommendations:")
    for i, rec in enumerate(plan.get("recommendations", []), 1):
        lines.append(f"{i}. {rec.get('title', '')}")
        lines.append(f"   Evidence: {', '.join(rec.get('evidence_ids', []))}")
        lines.append(f"   Reason: {rec.get('reason', '')}")
        lines.append(f"   Action: {rec.get('action', '')}")
        lines.append("")
    return "\n".join(lines).strip()


# -----------------------
# 主入口
# -----------------------

def main():
    st.set_page_config(page_title="Thesis Coach", layout="wide")
    st.title("Thesis Coach")
    # ---- 全局 CSS：卡片 + 小按钮 + evidence 卡片 ----
    st.markdown("""
            <style>
            /* Evidence 网格卡片（右栏） */
            .ebcs-evid-card {
                border-radius: 12px;
                border: 1px solid #374151;
                padding: 0.6rem 0.7rem;
                margin-bottom: 0.6rem;
                font-size: 0.85rem;
            }
            .ebcs-evid-card-selected {
                border-color: #f97316;
                box-shadow: 0 0 0 1px rgba(249,115,22,0.8);
            }
            .ebcs-evid-title {
                font-weight: 600;
                margin-bottom: 0.25rem;
            }
            .ebcs-evid-meta {
                font-size: 0.75rem;
                color: #9ca3af;
            }
            /* 右侧顶部：Evidence 选择按钮区域（chips） */
            .ebcs-evid-pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-bottom: 0.75rem;
            }

            /* Streamlit button 外层容器：key 以 evid-pill- 开头 */
            div[class*="st-key-evid-pill-"] {
                display: inline-flex;
            }
            /* 所有 evidence pill 的基础样式 */
            div[class*="st-key-evid-pill-P"] button {
                border-radius: 999px;
                padding: 0.3rem 0.9rem;
                font-size: 0.8rem;
                border: none;
                background-color: #DDE5E1 !important;
                color:#3A4743 !important;
            }
            div[class*="st-key-evid-pill-T"] button {
                border-radius: 999px;
                padding: 0.3rem 0.9rem;
                font-size: 0.8rem;
                border: none;
                background-color: #EFE8E2 !important;
                color: #524941 !important;
            }

            /* hover 效果 */
            div[class*="st-key-evid-pill-"] button:hover {
                background: #e5e7eb;
            }

            /* 选中的 pill：primary */
            div[class*="st-key-evid-pill-"] button[data-testid="stBaseButton-primary"] {
                background: #111827 !important;;
                color: #f9fafb !important;;
            }

            /* 未选中的 pill：secondary */
            div[class*="st-key-evid-pill-"] button[data-testid="stBaseButton-secondary"] {
                background: #f5f5f4 !important;;
                color: #111827 !important;;
            }

            .ev-pill {
            display: inline-block;
            margin: 3px 4px 0 0;
            padding: 2px 6px;
            border-radius: 8px;
            font-size: 11px;
            background: #f3f4f6;
            color: #374151;
            border: 1px solid #e5e7eb;
        }

        /* doc title */
        .ev-pill.doc {
            background: #e5e7eb;
            color: #111827;
        }

        /* stage / mode */
        .ev-pill.stage, .ev-pill.mode {
            background: #e0f2fe;
            color: #0369a1;
        }

        /* role: rubric / precedent */
        .ev-pill.role-rubric {
            background: #ffedd5;
            color: #c2410c;
        }
        .ev-pill.role-precedent {
            background: #dbeafe;
            color: #1d4ed8;
        }

        /* risk colors */
        .ev-pill.risk-high {
            background: #fee2e2;
            color: #b91c1c;
        }
        .ev-pill.risk-medium {
            background: #fef9c3;
            color: #92400e;
        }
        .ev-pill.risk-low {
            background: #dcfce7;
            color: #166534;
        }

        /* helpful / score */
        .ev-pill.helpful {
            background:#ecfdf5;
            color:#065f46;
        }
        .ev-pill.score {
            background:#f5f3ff;
            color:#5b21b6;
        }

        /* tags */
        .ev-pill.tag {
            background:#f3f4f6;
            color:#6b7280;
        }
        /* Target Streamlit Dialog container */
            div[role="dialog"][aria-label="dialog"] {
        width: 90% !important;           /* or 800px / 100rem / etc */
        max-width:1300px;
        border-radius: 12px !important;
        padding: 0 !important;
        margin: 0 auto !important;       /* center horizontally */
    }

/* Style the <p> inside all radio labels */
label[data-testid="stWidgetLabel"] p {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.3rem !important;
}

            </style>
            """, unsafe_allow_html=True)
    # 先确保 session_state 里有 user_id / 其它状态
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None

    # 先跑登录页，如果没登录就直接 return，不加载后面的聊天 UI
    login_page()
    if not st.session_state.get("user_id"):
        return
    maybe_show_pre_survey()

    init_state()

    policy_repo, thesis_repo = load_repositories()

    # ⬇️ Show one-time intro popup/banner
    show_intro_banner()
    # 只有在点击 “Got it” 后（= show_intro=False）才显示聊天机器人 + evidence vault
    if not st.session_state.show_intro:

        # 统一的 evidence 索引（给左栏按钮 + 右栏卡片用）
        evidence_index = {e.id: e for e in st.session_state.evidence_cards}
        # last_plan: Dict[str, Any] | None = None

        # ✅ 根据状态决定是否有右侧列
        if st.session_state.get("show_evidence_panel", False):
            col_chat, col_evid = st.columns([2.3, 1.7])
        else:
            col_chat, = st.columns([1])
            col_evid = None

        # ---------- 左栏：聊天 ----------
        with col_chat:
            st.caption(
                "Example question: I’m designing an AR interface to guide warehouse pickers. Main users are novice pickers in large e-commerce warehouses. I mainly care about task completion time and error rate. I’m in the proposal stage with a rough RQ draft. Can you help me improve the research question and check if it fits the IDE proposal rubric?")
            st.caption(
                "Example question: I’m preparing my green-light plan for a mental-health chatbot for university students.Users: Dutch master’s students experiencing study stress.Metrics: engagement (return visits) and perceived support.I have a draft method (diary study + interviews).Can you use the green-light checklist to tell me what is missing?")
            # 展示历史对话
            last_plan: dict | None = None
            for i, msg in enumerate(st.session_state.messages):
                role = msg.get("role")
                if role == "user":
                    with st.chat_message("user"):
                        st.markdown(msg.get("content", ""))
                elif role == "assistant" and "plan" in msg:
                    plan_dict = msg["plan"]
                    last_plan = plan_dict
                    with st.chat_message("assistant"):
                        render_plan_as_cards(plan_dict, evidence_index, key_prefix=f"msg{i}_")
                else:
                    with st.chat_message("assistant"):
                        st.markdown(msg.get("content", ""))

            # 输入框：如果正在生成 或 已经有 last_plan，就不再显示输入框
            user_text = None
            if st.session_state.busy:
                st.info("The coach is thinking…")
            elif last_plan is not None:
                # 你也可以换成 st.empty() 什么都不显示；下面只是友好提示
                if not st.session_state.get("post_survey_done", False):
                    if st.button("End this conversation & fill post-survey"):
                        post_survey_dialog()
                else:
                    st.info("Thank you for the participating!")
            else:
                user_text = st.chat_input("Describe your situation / question (press Enter to send)")
            if user_text:
                # 标记忙碌（下一轮就不会显示输入框）
                st.session_state.busy = True
                user_id = st.session_state.get("user_id")
                # ✅ 记录这条触发 plan 的问题时间
                st.session_state["last_question_ts"] = datetime.utcnow().isoformat()

                # 记录用户消息
                st.session_state.messages.append({"role": "user", "content": user_text})
                # 汇总上下文
                user_msgs = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
                task_context = "\n".join(user_msgs)
                st.session_state.task_context = task_context

                # 路由 & 追问逻辑（Stage×Mode×Gap + AI 生成 follow-up）
                align = st.session_state.alignment
                if not align.get("enough_info", False):
                    # 🔄 在「路由 + 可能追问」阶段也给一点可视化进度
                    with st.status("The coach is checking what is still unclear…", expanded=True) as follow_status:
                        follow_status.update(
                            label="Step 1/2: Checking how well your project fits the rubrics…",
                            state="running",
                        )
                        follow_status.write("Embedding your notes and routing to the right stage/mode…")
                        routing = route_and_maybe_ask(task_context)

                        st.session_state.alignment.update(
                            {
                                "stage": routing.get("stage", align["stage"]),
                                "mode": routing.get("mode", align["mode"]),
                                "gap": routing.get("gap", align["gap"]),
                                "enough_info": routing.get("enough_info", False),
                            }
                        )
                        follow_status.write(
                            f"You are on {routing.get('stage', align['stage'])}, {routing.get('mode', align['mode'])}, {routing.get('mode', align['mode'])}")
                        count = st.session_state.followup_count
                        MAX_FOLLOWUP = 5

                        if not routing["enough_info"]:
                            # 需要继续补信息 → 生成澄清问题
                            follow_status.update(
                                label="Step 2/2: Not enough context, writing a focused follow-up question for you…",
                                state="running",
                            )

                            if count >= MAX_FOLLOWUP:
                                # 走 exhausted_warning 分支
                                warn_text = build_exhausted_warning(
                                    task_context=task_context,
                                    alignment=st.session_state.alignment,
                                    followup_count=count,
                                )
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": warn_text}
                                )
                                st.session_state.alignment["enough_info"] = True
                                follow_status.update(
                                    label="Done — using generic but reasonable assumptions now ✅",
                                    state="complete",
                                )
                            else:
                                # 正常 follow-up question 分支
                                st.session_state.followup_count = count + 1
                                follow_q = build_followup_question(
                                    task_context=task_context,
                                    routing=routing,
                                    followup_count=count,
                                )
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": follow_q}
                                )
                                follow_status.update(
                                    label="Done — I have a clarifying question for you ✅",
                                    state="complete",
                                )
                                # 结束 loading，允许用户继续输入
                                st.session_state.busy = False
                                st.rerun()
                                return
                        else:
                            # 信息已经够了，重置追问计数，后面走 4 步的 plan 生成流程
                            st.session_state.followup_count = 0
                            follow_status.update(
                                label="Enough context — moving on to evidence retrieval ✅",
                                state="complete",
                            )

                # enough_info=True → 检索 + 生成 plan
                with st.status("Generating coach recommendations…", expanded=True) as status:
                    # Step 1: embed + 对齐
                    status.update(label="Step 1/4: Embedding the conversation and aligning Stage × Mode × Gap…",
                                  state="running")
                    q_emb = embed_text(task_context)
                    align = st.session_state.alignment

                    # Step 2: RAG retrieval
                    status.update(label="Step 2/4: Retrieving evidence from rubrics and past theses…", state="running")
                    cards = fuse_evidence(
                        query_text=task_context,
                        query_emb=q_emb,
                        stage=align["stage"],
                        mode=align["mode"],
                        gap=align["gap"],
                        policy_repo=policy_repo,
                        thesis_repo=thesis_repo,
                    )
                    st.session_state.evidence_cards = cards

                    # ✅ 本轮 round index：生成 plan 前先 +1
                    st.session_state.round_id += 1
                    round_idx = st.session_state.round_id

                    # Step 3: Generate structured action plan
                    status.update(label="Step 3/4: Generating a structured action plan from evidence…", state="running")
                    plan_obj = generate_coach_plan(
                        user_input=user_text,
                        stage=align["stage"],
                        mode=align["mode"],
                        gap=align["gap"],
                        task_context=task_context,
                        evidence_cards=cards,
                        history=st.session_state.messages,
                    )

                    # Save the assistant message containing the plan
                    st.session_state.messages.append(
                        {"role": "assistant", "plan": plan_obj.model_dump()}
                    )

                    # ✅ 记录回答时间 + 写 chat_turns_ebcs.csv
                    ts_q = st.session_state.get("last_question_ts", datetime.utcnow().isoformat())
                    ts_a = datetime.utcnow().isoformat()
                    question_text = st.session_state.get("last_question_text", user_text)

                    log_chat_turn(
                        user_id=st.session_state.get("user_id"),
                        round_index=round_idx,
                        timestamp_question=ts_q,
                        timestamp_answer=ts_a,
                        question=question_text,
                        plan_obj=plan_obj,
                        evidence_cards=cards,
                        alignment=align,
                    )

                    status.update(label="Step 4/4: Done — coach recommendations generated ✅", state="complete")

                # End generation; clear busy flag
                st.session_state.busy = False
                st.rerun()

            # ---------- 右栏：Evidence 卡片网格 + 导出 plan ----------
            if col_evid is not None:
                print(f"2 col_evid {col_evid}")
                with col_evid:
                    st.markdown("""
                            <style>
                            @keyframes slideInRight {
                                from { transform: translateX(100%); opacity: 0; }
                                to   { transform: translateX(0);   opacity: 1; }
                            }
                            .ebcs-evid-panel-anim {
                                animation: slideInRight 0.25s ease-out;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                    st.markdown('<div class="ebcs-evid-panel-anim">', unsafe_allow_html=True)
                    st.subheader("Evidence Vault")
                    st.caption("Rubrics & Precedents (Click the left / right buttons to highlight)")

                    evids = st.session_state.evidence_cards
                    selected_eid = st.session_state.get("selected_evidence")

                    if evids:
                        # # # 如果还没有选中的 evidence，默认选第一条
                        # if not selected_eid:
                        #     selected_eid = evids[0].id
                        #     st.session_state.selected_evidence = selected_eid
                        #
                        # # ---------- 顶部：小圆角按钮（只显示标题） ----------
                        # st.markdown('<div class="ebcs-evid-pill-row">', unsafe_allow_html=True)
                        # pill_row = st.container()
                        # with pill_row:
                        #     for e in evids:
                        #         label = f"[{e.id}] {e.title}"  # e.g. [P1] Public presentation standards (rubric)
                        #         is_sel = (e.id == selected_eid)
                        #         btn_type = "primary" if is_sel else "secondary"
                        #
                        #         if st.button(label, key=f"evid-pill-{e.id}", type=btn_type):
                        #             st.session_state.selected_evidence = e.id
                        #             selected_eid = e.id  # 当前循环内同步更新
                        #
                        # st.markdown("</div>", unsafe_allow_html=True)

                        # ---------- 下方：只展示选中的完整 card ----------
                        sel = next((ev for ev in evids if ev.id == selected_eid), None)
                        if sel is not None:
                            # meta_str = "; ".join(
                            #     f"{k}: {v}" for k, v in sel.meta.items() if v
                            # )
                            # turn meta into pills
                            def pill(label, value, cls=""):
                                return f'<span class="ev-pill {cls}"><b>{label}</b>: {value}</span>'

                            meta_pills = []

                            if "doc_title" in sel.meta:
                                meta_pills.append(pill("doc", sel.meta["doc_title"], "doc"))

                            if "stage" in sel.meta:
                                meta_pills.append(pill("stage", sel.meta["stage"], "stage"))

                            if "mode" in sel.meta:
                                meta_pills.append(pill("mode", sel.meta["mode"], "mode"))

                            if "risk" in sel.meta:
                                risk = sel.meta["risk"]
                                cls = "risk-high" if risk == "high" else "risk-medium" if risk == "medium" else "risk-low"
                                meta_pills.append(pill("risk", risk, cls))

                            if "llm_role" in sel.meta:
                                role = sel.meta["llm_role"]
                                cls = "role-rubric" if role == "rubric" else "role-precedent"
                                meta_pills.append(pill("role", role, cls))

                            if "helpful" in sel.meta:
                                meta_pills.append(pill("helpful", sel.meta["helpful"], "helpful"))

                            if "score" in sel.meta:
                                meta_pills.append(pill("score", sel.meta["score"], "score"))

                            if "tags" in sel.meta:
                                tags = sel.meta["tags"].split(",")
                                for t in tags:
                                    meta_pills.append(pill("tag", t.strip(), "tag"))

                            meta_html = "".join(meta_pills)

                            # 把 snippet 从 markdown 转成 HTML
                            snippet_html = md.markdown(sel.snippet)
                            html = f"""
                            <div class="ebcs-evid-card ebcs-evid-card-selected">
                              <div class="ebcs-evid-title">[{sel.id}] {sel.title}</div>
                              <div class="ebcs-evid-snippet">{snippet_html}</div>
                              <!-- metadata pills row -->
                              <div class="ebcs-evid-meta">
                                  {meta_html}
                              </div>
                            </div>
                            """
                            st.markdown(html, unsafe_allow_html=True)
                            # ==== 新增：原始文件 section ====
                            source_path = sel.meta.get("source_path")
                            source_chunk_md = sel.meta.get("source_chunk_md")
                            # st.write("DEBUG meta for", sel.id, ":", sel.meta)
                            if source_chunk_md or source_path:
                                # full_md = load_markdown_file(source_path)
                                # 在渲染前
                                with st.expander("Show raw text excerpts (you can scroll down in the box to see more)"):
                                    user_id = st.session_state.get("user_id")
                                    round_idx = st.session_state.get("round_id", 0)
                                    src_flag_key = f"logged_source_expand_{round_idx}_{sel.id}"

                                    if user_id and not st.session_state.get(src_flag_key, False):
                                        log_evidence_event(
                                            user_id=user_id,
                                            round_index=round_idx,
                                            event_type="source_section_expand",
                                            evid=sel,
                                            rec_index=None,
                                            rec_title=None,
                                            extra={"source_path": source_path},
                                        )
                                        st.session_state[src_flag_key] = True
                                    if source_chunk_md:
                                        import re
                                        fixed_md = re.sub(r"(?m)^\s*#+", "###", source_chunk_md)
                                        import markdown
                                        fixed_html = markdown.markdown(fixed_md, extensions=["tables"])
                                        html = f"""
                                        <div style="max-height:30rem; overflow-y:auto; padding-right:12px;">
                                        {fixed_html}
                                        </div>
                                        """
                                        st.markdown(html, unsafe_allow_html=True)
                                    else:
                                        st.caption("Original file could not be loaded or section not found.")
                    else:
                        st.caption("No rubrics or precedents have been retrieved for this round yet.")

                    st.markdown("</div>", unsafe_allow_html=True)
                    # 导出当前行动计划
                    # 假设 last_plan 不为 None 时，说明本轮聊天已经有一个完整的推荐
                    # if last_plan:
                    #     if not st.session_state.get("post_survey_done", False):
                    #         if st.button("End this conversation"):
                    #             post_survey_dialog()
                    #     else:
                    #         st.success("You've finished the survey, thank you！")

                    #     st.markdown("---")
                    #     st.caption("Export the action plan for this round (copy it to your log / planning sheet).")
                    #     export_text = build_export_text(last_plan)
                    #     st.text_area("Plan", export_text, height=220)

    #
    # # ---------- Debug：内部 Stage×Mode×Gap 状态 ----------
    with st.expander("Debug：内部 Stage × Mode × Gap 状态（开发用）"):
        st.json(st.session_state.alignment)


if __name__ == "__main__":
    main()