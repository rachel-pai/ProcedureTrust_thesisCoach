# baseline_chat.py
import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import csv
from datetime import datetime
import markdown as md   # 👈 新增
import re
load_dotenv()

# 模型配置
OPENAI_MODEL = os.getenv("EBCS_MODEL", "gpt-5-mini")
EMBED_MODEL = os.getenv("EBCS_EMBED_MODEL", "text-embedding-3-large")

client = OpenAI()

# # Baseline VS 路径
# VS_DIR = Path("baseline_from_indexes_vs")
# ENTRIES_PATH = VS_DIR / "baseline_from_idx_entries.json"
# EMB_PATH = VS_DIR / "baseline_from_idx_embeddings.npy"

# -----------------------
# 日志 & 问卷 CSV
# -----------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
PRE_FILE = LOG_DIR / "pre_survey_baseline.csv"
POST_FILE = LOG_DIR / "post_survey_baseline.csv"
# 👇 新增：聊天轮次 & snippet 点击日志
CHAT_LOG_FILE = LOG_DIR / "chat_turns_baseline.csv"
SNIPPET_LOG_FILE = LOG_DIR / "snippet_clicks_baseline.csv"


from qdrant_client import QdrantClient

# -----------------------
# Qdrant 配置（Baseline 版）
# -----------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# 这个 collection 名字你按自己在 Qdrant 里建的来改
BASELINE_COLLECTION = os.getenv("QDRANT_BASELINE_COLLECTION")

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )


def append_csv_row(path: Path, fieldnames, row_dict):
    """Append one row to a CSV (create header if file doesn’t exist)."""
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


# -----------------------
# 登录逻辑
# -----------------------

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

# =======
# for the raw snippt, change the url path so that the images can be shown
# =====
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


# -----------------------
# Pre-survey Dialog（radio 1–7）
# -----------------------
# -----------------------
# Pre-survey Dialog（radio 1–7）
# -----------------------
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
            # 原有字段
            "prior_exp_llm": 3,
            "prior_trust": 4,
            "topic_clarity": 3,
            "rq_confidence": 3,
            "open_expectations": "",
            # 新增：背景 + 自我效能 + 对流程的态度
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

    # --- 1–7 频率量表 ---
    freq_scale = {
        1: "Very rarely / almost never",
        2: "Rarely",
        3: "Sometimes",
        4: "About half of the time",
        5: "Often",
        6: "Very often",
        7: "Almost always"
    }

    # --- 1–7 “程度”量表（信任、自信、清晰度等）---
    degree_scale = {
        1: "Very low",
        2: "Low",
        3: "Somewhat low",
        4: "Neutral / in the middle",
        5: "Somewhat high",
        6: "High",
        7: "Very high"
    }

    # --- rubric 熟悉度 1–4 量表 ---
    rubric_scale = {
        1: "I have never seen the official IDE thesis rubrics / checklists",
        2: "I have heard of them but never looked at them in detail",
        3: "I have looked at them a few times",
        4: "I often refer to them when working on my thesis"
    }

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
    )
    st.session_state.pre_tmp["stage"] = stage

    domain_default = pre_tmp.get("domain_short", "")
    domain_short = st.text_input(
        "In one or two short phrases, how would you describe your graduation topic or domain? (e.g., hospital robot collaboration, warehouse teamwork, etc.)",
        value=domain_default,
    )
    st.session_state.pre_tmp["domain_short"] = domain_short

    rubric_familiarity_default = pre_tmp.get("rubric_familiarity", 2)
    rubric_familiarity = st.radio(
        "Before using this system, how familiar are you with the official IDE graduation rubrics or stage checklists?",
        options=list(rubric_scale.keys()),
        # index=max(0, min(3, rubric_familiarity_default - 1)),
        index=None,
        format_func=lambda x: rubric_scale[x],
        horizontal=True,
    )
    st.session_state.pre_tmp["rubric_familiarity"] = rubric_familiarity

    # =========================
    #  B. 之前对 LLM 的经验 & 信任
    # =========================
    st.subheader("B. Prior experience with AI tools")

    prior_exp_llm_default = pre_tmp.get("prior_exp_llm", 3)
    prior_exp_llm = st.radio(
        "How often have you previously used large language models (e.g., ChatGPT) for study or academic work?",
        options=list(freq_scale.keys()),
        # index=max(0, min(6, prior_exp_llm_default - 1)),
        index=None,
        format_func=lambda x: freq_scale[x],
        horizontal=True
    )
    st.session_state.pre_tmp["prior_exp_llm"] = prior_exp_llm

    prior_trust_default = pre_tmp.get("prior_trust", 4)
    prior_trust = st.radio(
        "Before using this system, how much do you trust AI-based feedback or supervision tools?",
        options=list(degree_scale.keys()),
        # index=max(0, min(6, prior_trust_default - 1)),
        index=None,
        format_func=lambda x: degree_scale[x],
        horizontal=True
    )
    st.session_state.pre_tmp["prior_trust"] = prior_trust

    # =========================
    #  C. 对课题清晰度 & 自我效能
    # =========================
    st.subheader("C. Your thesis clarity and confidence")

    topic_clarity_default = pre_tmp.get("topic_clarity", 3)
    topic_clarity = st.radio(
        "How clear are you about your graduation project topic or research direction?",
        options=list(degree_scale.keys()),
        # index=max(0, min(6, topic_clarity_default - 1)),
        index=None,
        format_func=lambda x: degree_scale[x],
        horizontal=True
    )
    st.session_state.pre_tmp["topic_clarity"] = topic_clarity

    rq_confidence_default = pre_tmp.get("rq_confidence", 3)
    rq_confidence = st.radio(
        "How confident are you right now about your research question or thesis plan?",
        options=list(degree_scale.keys()),
        # index=max(0, min(6, rq_confidence_default - 1)),
        index=None,
        format_func=lambda x: degree_scale[x],
        horizontal=True
    )
    st.session_state.pre_tmp["rq_confidence"] = rq_confidence

    rq_self_efficacy_default = pre_tmp.get("rq_self_efficacy", 3)
    rq_self_efficacy = st.radio(
        "Without any special tools, how confident are you that you can formulate a good research question (RQ)?",
        options=list(degree_scale.keys()),
        # index=max(0, min(6, rq_self_efficacy_default - 1)),
        index=None,
        format_func=lambda x: degree_scale[x],
        horizontal=True,
    )
    st.session_state.pre_tmp["rq_self_efficacy"] = rq_self_efficacy

    method_self_efficacy_default = pre_tmp.get("method_self_efficacy", 3)
    method_self_efficacy = st.radio(
        "Without any special tools, how confident are you in choosing suitable methods and measurements to support your RQ?",
        options=list(degree_scale.keys()),
        # index=max(0, min(6, method_self_efficacy_default - 1)),
        index=None,
        format_func=lambda x: degree_scale[x],
        horizontal=True,
    )
    st.session_state.pre_tmp["method_self_efficacy"] = method_self_efficacy

    rubric_eval_knowledge_default = pre_tmp.get("rubric_eval_knowledge", 3)
    rubric_eval_knowledge = st.radio(
        "How well do you think you understand how IDE assessors will judge whether a thesis plan is ‘ready’ based on the official criteria?",
        options=list(degree_scale.keys()),
        # index=max(0, min(6, rubric_eval_knowledge_default - 1)),
        index=None,
        format_func=lambda x: degree_scale[x],
        horizontal=True,
    )
    st.session_state.pre_tmp["rubric_eval_knowledge"] = rubric_eval_knowledge

    # =========================
    #  D. 对“按流程来”的态度（procedural orientation）
    # =========================
    st.subheader("D. Your attitude towards structured processes")

    procedural_preference_default = pre_tmp.get("procedural_preference", 4)
    procedural_preference = st.radio(
        "In your study or projects, how much do you prefer having clear steps and checkpoints, instead of full freedom?",
        options=list(degree_scale.keys()),
        # index=max(0, min(6, procedural_preference_default - 1)),
        index=None,
        format_func=lambda x: degree_scale[x],
        horizontal=True,
    )
    st.session_state.pre_tmp["procedural_preference"] = procedural_preference

    procedural_acceptance_default = pre_tmp.get("procedural_acceptance", 4)
    procedural_acceptance = st.radio(
        "If a digital tool requires you to follow a certain process (e.g., read rubrics, check evidence) before continuing, how acceptable is that to you?",
        options=list(degree_scale.keys()),
        # index=max(0, min(6, procedural_acceptance_default - 1)),
        index=None,
        format_func=lambda x: degree_scale[x],
        horizontal=True,
    )
    st.session_state.pre_tmp["procedural_acceptance"] = procedural_acceptance

    # =========================
    #  E. 开放性问题：对系统的期待
    # =========================
    st.subheader("E. Expectations for the Thesis Coach")

    open_expectations_default = pre_tmp.get("open_expectations", "")
    open_expectations = st.text_area(
        "What do you expect the Thesis Coach to help you with? (optional, you can answer in English or Chinese)",
        open_expectations_default,
    )
    st.session_state.pre_tmp["open_expectations"] = open_expectations

    # —— Submit ——
    if st.button("Submit", type="primary"):
        missing = []

        if None in (stage, domain_short, rubric_familiarity,prior_exp_llm,prior_trust,topic_clarity,rq_confidence,
                    rq_self_efficacy,method_self_efficacy,rubric_eval_knowledge,procedural_preference,procedural_acceptance):
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
            # LLM & trust
            "prior_exp_llm": prior_exp_llm,
            "prior_trust": prior_trust,
            # clarity & efficacy
            "topic_clarity": topic_clarity,
            "rq_confidence": rq_confidence,
            "rq_self_efficacy": rq_self_efficacy,
            "method_self_efficacy": method_self_efficacy,
            "rubric_eval_knowledge": rubric_eval_knowledge,
            # procedural attitude
            "procedural_preference": procedural_preference,
            "procedural_acceptance": procedural_acceptance,
            # open text
            "open_expectations": open_expectations,
        }
        append_csv_row(PRE_FILE, fieldnames=list(row.keys()), row_dict=row)

        st.success("Thank you! You may now use the system.")
        st.rerun()


def maybe_show_pre_survey():
    if st.session_state.get("pre_survey_done"):
        return
    pre_survey_dialog()


# -----------------------
# Post-survey Dialog（radio 1–7）
# -----------------------
# -----------------------
# Post-survey Dialog（radio 1–7）
# -----------------------
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
            # 原有字段
            "perceived_usefulness": 4,
            "perceived_procedural_fairness": 4,
            "perceived_transparency": 4,
            "trust_after": 4,
            "clarity_improved": 4,
            "cognitive_load": 4,
            "satisfaction": 4,
            "open_feedback": "",
            # 新增：更细的 procedural trust / 证据使用 / 校准 / 可用性 / UI
            "procedural_rules_clarity": 4,
            "procedural_predictability": 4,
            "procedural_voice": 4,
            "evidence_engagement": 4,
            "evidence_cross_check": 4,
            "safety_support": 4,
            "trust_double_check": 4,
            "overtrust_concern": 4,
            "usability_ease": 4,
            "helpful_elements": "",
        }

    post_tmp = st.session_state.post_tmp

    degree_scale = {
        1: "Very low / very negative",
        2: "Low",
        3: "Somewhat low",
        4: "Neutral / in the middle",
        5: "Somewhat high",
        6: "High",
        7: "Very high / very positive"
    }
    demand_scale = {
        1: "Not demanding at all",
        2: "Slightly demanding",
        3: "Somewhat demanding",
        4: "Moderately demanding",
        5: "Quite demanding",
        6: "Very demanding",
        7: "Extremely demanding"
    }

    def radio_1to7(label: str, field: str, scale: Dict[int, str]) -> int:
        default_val = post_tmp.get(field, 4)
        return st.radio(
            label,
            options=list(scale.keys()),
            # index=max(0, min(6, default_val - 1)),
            index=None,
            format_func=lambda x: scale[x],
            horizontal=True
        )

    # =========================
    #  A. Overall usefulness, clarity, trust, satisfaction
    # =========================
    st.subheader("A. Overall experience")

    perceived_usefulness = radio_1to7(
        "Overall, how useful was the Thesis Coach for your current thesis task?",
        "perceived_usefulness",
        degree_scale,
    )
    st.session_state.post_tmp["perceived_usefulness"] = perceived_usefulness

    clarity_improved = radio_1to7(
        "Did the session help you become clearer about your thesis problem, RQ, or next steps?",
        "clarity_improved",
        degree_scale,
    )
    st.session_state.post_tmp["clarity_improved"] = clarity_improved

    trust_after = radio_1to7(
        "After using this system, how much do you trust its feedback and guidance?",
        "trust_after",
        degree_scale,
    )
    st.session_state.post_tmp["trust_after"] = trust_after

    satisfaction = radio_1to7(
        "Overall, how satisfied are you with this interaction with the Thesis Coach?",
        "satisfaction",
        degree_scale,
    )
    st.session_state.post_tmp["satisfaction"] = satisfaction

    usability_ease = radio_1to7(
        "Overall, how easy or difficult was it to use the system to complete this task?",
        "usability_ease",
        degree_scale,
    )
    st.session_state.post_tmp["usability_ease"] = usability_ease

    cognitive_load = radio_1to7(
        "How mentally demanding did you find the interaction with the system?",
        "cognitive_load",
        demand_scale,
    )
    st.session_state.post_tmp["cognitive_load"] = cognitive_load

    # =========================
    #  B. Procedural fairness & transparency (更细拆分)
    # =========================
    st.subheader("B. Procedural fairness and transparency")

    perceived_procedural_fairness = radio_1to7(
        "Did the process of getting feedback from the system feel systematic and fair (e.g., based on clear criteria rather than arbitrary answers)?",
        "perceived_procedural_fairness",
        degree_scale,
    )
    perceived_transparency = radio_1to7(
        "How transparent did the system feel about *why* it gave particular suggestions (e.g., showing rubrics or precedents)?",
        "perceived_transparency",
        degree_scale,
    )

    procedural_rules_clarity = radio_1to7(
        "During this session, how clear did it feel that the system was following a consistent set of rules or criteria?",
        "procedural_rules_clarity",
        degree_scale,
    )

    procedural_predictability = radio_1to7(
        "How predictable did the system’s next steps feel (e.g., what it would ask you to do next)?",
        "procedural_predictability",
        degree_scale,
    )

    procedural_voice = radio_1to7(
        "When you did not fully agree with the system’s suggestions, did you still feel you had room to express your own ideas or choose another path?",
        "procedural_voice",
        degree_scale,
    )

    # =========================
    #  C. Evidence use and perceived safety
    # =========================
    st.subheader("C. Evidence use and perceived safety")

    evidence_engagement = radio_1to7(
        "This system encouraged me to actually open and read the sources or snippets it provided (instead of just trusting the answer).",
        "evidence_engagement",
        degree_scale,
    )

    evidence_cross_check = radio_1to7(
        "Before making decisions, I usually checked whether at least one or two snippets really supported the system’s suggestions.",
        "evidence_cross_check",
        degree_scale,
    )

    safety_support = radio_1to7(
        "In this task, I felt that the system helped me avoid decisions that might be risky or weak for my thesis.",
        "safety_support",
        degree_scale,
    )

    # =========================
    #  D. Trust calibration（不过度依赖 vs 合理检查）
    # =========================
    st.subheader("D. Trust calibration")

    trust_double_check = radio_1to7(
        "Even when the system sounded confident, I still used my own judgement or other information to double-check important suggestions.",
        "trust_double_check",
        degree_scale,
    )

    overtrust_concern = radio_1to7(
        "At some moments in this task, I felt that I might be relying too much on the system.",
        "overtrust_concern",
        degree_scale,
    )


    # =========================
    #  E. Specific interface elements
    # =========================
    st.subheader("E. Interface elements")

    helpful_elements_options = [
        "The separate sources / snippet buttons",
        "The right-hand snippet panel with raw text",
        "Seeing titles and similarity scores of retrieved snippets",
        "The general chat interface",
        "Other elements (please describe in the text box below)",
        "None of the above were particularly helpful",
    ]
    # Load selection safely from session_state (must be list, not string)
    if isinstance(post_tmp.get("helpful_elements"), list):
        default_list = post_tmp["helpful_elements"]
    else:
        default_list = []

    helpful_elements_selected = st.multiselect(
        "Which interface elements of the Thesis Coach did you personally find especially helpful in this session? (you can select multiple)",
        options=helpful_elements_options,
        default=default_list,
    )

    open_feedback_default = post_tmp.get("open_feedback (optional)", "")
    open_feedback = st.text_area(
        "If you have any comments about what worked well or what felt problematic (e.g., fairness, clarity, missing support), please write them here.",
        open_feedback_default,
    )

    # —— Submit ——
    if st.button("Submit", type="primary"):
        missing = []

        if None in (perceived_procedural_fairness, perceived_transparency,
                    procedural_rules_clarity,procedural_predictability,procedural_voice,evidence_engagement,
                    evidence_cross_check,safety_support,trust_double_check,overtrust_concern):
            missing.append("Some required fields were not answered.")

        if missing:
            st.error("You must answer all required questions:\n- " + "\n- ".join(missing))
            st.stop()

        # DO NOT write back immediately — only write back when submit
        st.session_state.post_tmp["helpful_elements"] = helpful_elements_selected
        st.session_state.post_tmp["open_feedback"] = open_feedback
        st.session_state.post_tmp["perceived_procedural_fairness"] = perceived_procedural_fairness
        st.session_state.post_tmp["perceived_transparency"] = perceived_transparency
        st.session_state.post_tmp["procedural_rules_clarity"] = procedural_rules_clarity
        st.session_state.post_tmp["procedural_predictability"] = procedural_predictability
        st.session_state.post_tmp["procedural_voice"] = procedural_voice
        st.session_state.post_tmp["evidence_engagement"] = evidence_engagement
        st.session_state.post_tmp["evidence_cross_check"] = evidence_cross_check
        st.session_state.post_tmp["safety_support"] = safety_support
        st.session_state.post_tmp["trust_double_check"] = trust_double_check
        st.session_state.post_tmp["overtrust_concern"] = overtrust_concern
        st.session_state["post_survey"] = st.session_state.post_tmp.copy()
        st.session_state["post_survey_done"] = True

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            # overall
            "perceived_usefulness": perceived_usefulness,
            "clarity_improved": clarity_improved,
            "trust_after": trust_after,
            "satisfaction": satisfaction,
            "usability_ease": usability_ease,
            "cognitive_load": cognitive_load,
            # procedural trust / transparency
            "perceived_procedural_fairness": perceived_procedural_fairness,
            "perceived_transparency": perceived_transparency,
            "procedural_rules_clarity": procedural_rules_clarity,
            "procedural_predictability": procedural_predictability,
            "procedural_voice": procedural_voice,
            # evidence & safety
            "evidence_engagement": evidence_engagement,
            "evidence_cross_check": evidence_cross_check,
            "safety_support": safety_support,
            # calibration
            "trust_double_check": trust_double_check,
            "overtrust_concern": overtrust_concern,
            # UI elements & open feedback
            "helpful_elements": "; ".join(helpful_elements_selected),
            "open_feedback": open_feedback,
        }
        append_csv_row(POST_FILE, fieldnames=list(row.keys()), row_dict=row)

        st.success("Thank you for your feedback!")
        st.rerun()

# -----------------------
# 工具函数（向量检索）
# -----------------------
# def load_vector_store():
#     if not ENTRIES_PATH.exists() or not EMB_PATH.exists():
#         raise FileNotFoundError(
#             f"Vector store not found. Please run build_baseline_from_indexes.py first.\n"
#             f"Expected files:\n  {ENTRIES_PATH}\n  {EMB_PATH}"
#         )
#     entries = json.loads(ENTRIES_PATH.read_text(encoding="utf-8"))
#     emb_matrix = np.load(EMB_PATH)
#     if emb_matrix.dtype != np.float32:
#         emb_matrix = emb_matrix.astype("float32")
#     return entries, emb_matrix


# @st.cache_resource
# def get_vs_cached():
#     return load_vector_store()


def embed_text(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
    )
    return np.array(resp.data[0].embedding, dtype="float32")

def cosine_sim_matrix(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    denom = (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec) + 1e-9)
    return (matrix @ query_vec) / denom

def retrieve_top_k(
    query: str,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    在 Qdrant 的 BASELINE_COLLECTION 里做向量搜索，
    返回结构尽量保持和原来 entries+emb_matrix 版本一致。
    """
    q_vec = embed_text(query)
    client_q = get_qdrant_client()

    hits = client_q.search(
        collection_name=BASELINE_COLLECTION,
        query_vector=q_vec,
        limit=top_k,
        with_payload=True,
    )

    results: List[Dict[str, Any]] = []
    for rank, h in enumerate(hits, start=1):
        p = h.payload or {}
        # 兼容不同 payload 命名
        results.append(
            {
                "rank": rank,
                "score": float(h.score or 0.0),
                # 原 baseline entries 里是 "id"；如果你导入 Qdrant 时用的是 "raw_id"，这里兜底一下
                "id": p.get("id") or p.get("raw_id") or h.id,
                "source_type": p.get("source_type"),
                "doc_title": p.get("doc_title"),
                # baseline UI 下面会拿 source_id 去修图片链接，所以这里保留
                "source_id": p.get("source_id") or p.get("source_path") or "",
                "source_path": p.get("source_path"),
                # 文本字段：你在 Qdrant 里可以叫 "text"、"source_chunk_md" 或 "raw_excerpt_md"
                "text": p.get("text")
                        or p.get("source_chunk_md")
                        or p.get("raw_excerpt_md")
                        or "",
            }
        )
    return results

def call_llm_with_context(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    context_blocks = []
    for r in retrieved_chunks:
        src = r["source_type"]
        doc = r.get("doc_title") or r.get("source_id") or "(unknown)"
        header = f"[{r['rank']}] ({src}) {doc}"
        block = f"{header}\n{r['text']}"
        context_blocks.append(block)
    context_text = "\n\n---\n\n".join(context_blocks)
    system_prompt = (
        "You are a simple RAG-based thesis coach for IDE MSc graduation projects.\n"
        "You must answer the student's question ONLY using the provided context snippets, which come from official rubrics and past IDE master theses.\n"
        "If the answer is not clearly supported by the context, say that you cannot be sure "
        "and explain what additional information or documents the student should check.\n"
        "Be concrete and concise. You may answer in the same language as the question (Chinese/English)."
    )
    user_prompt = (
        "Student question:\n"
        f"{question}\n\n"
        "Relevant context snippets:\n"
       "---------------------------------\n"
        f"{context_text}\n"
        "---------------------------------\n"
        "Now answer the student's question based on these snippets."
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
    )

    if hasattr(resp, "output_text"):
        return resp.output_text.strip()

    texts = []
    for o in resp.output:
        for c in o.content:
            if getattr(c, "type", "") == "output_text":
                texts.append(c.text)
    return "".join(texts).strip()


# -----------------------
# 状态管理 & UI
# -----------------------

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retrievals" not in st.session_state:
        st.session_state.retrievals = []
    if "busy" not in st.session_state:
        st.session_state.busy = False
    # 新增：右侧 snippet 面板控制
    if "show_snippet_panel" not in st.session_state:
        st.session_state.show_snippet_panel = False
    if "selected_snippet_rank" not in st.session_state:
        st.session_state.selected_snippet_rank = None
        # 👇 新增：本 session 内第几轮问答（0,1,2,...）
    if "turn_counter" not in st.session_state:
        st.session_state.turn_counter = 0

def log_snippet_click(user_id: str, turn_index: int, snippet: Dict[str, Any], action: str):
    """
    记录每次点击 snippetX 的行为：
    - user_id: 当前用户
    - turn_index: 第几轮问答
    - snippet: retrieve_top_k 返回的那一条 dict
    - action: 'show' or 'hide'
    """
    try:
        row = {
            "timestamp_click": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "turn_index": turn_index,
            "action": action,  # show / hide
            "rank": snippet.get("rank"),
            "snippet_id": snippet.get("id"),
            "source_type": snippet.get("source_type"),
            "score": snippet.get("score"),
            "doc_title": snippet.get("doc_title") or snippet.get("source_id") or "",
        }
        append_csv_row(SNIPPET_LOG_FILE, fieldnames=list(row.keys()), row_dict=row)
    except Exception as e:
        print("Failed to log snippet click:", e)


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
def toggle_snippet_panel(rank: int):
    prev = st.session_state.get("selected_snippet_rank")
    showing = st.session_state.get("show_snippet_panel", False)
    if showing and prev == rank:
        st.session_state.show_snippet_panel = False
        st.session_state.selected_snippet_rank = None
    else:
        st.session_state.show_snippet_panel = True
        st.session_state.selected_snippet_rank = rank


def main():
    st.set_page_config(page_title="Baseline RAG Thesis Coach", layout="wide")
    st.title("Baseline A · Simple RAG Thesis Coach")
    st.markdown("""
            <style>
        /* Target Streamlit Dialog container */
            div[role="dialog"][aria-label="dialog"] {
        width: 90% !important;           /* or 800px / 100rem / etc */
        max-width:1300px;
        border-radius: 12px !important;
        padding: 0 !important;
        margin: 0 auto !important;       /* center horizontally */
    }

            </style>
            """, unsafe_allow_html=True)
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None

    login_page()
    if not st.session_state.get("user_id"):
        return

    maybe_show_pre_survey()
    init_state()

    # try:
    #     entries, emb_matrix = get_vs_cached()
    # except FileNotFoundError as e:
    #     st.error(str(e))
    #     st.stop()

    # ⬇️ Show one-time intro popup/banner
    show_intro_banner()
    # 只有在点击 “Got it” 后（= show_intro=False）才显示聊天机器人 + evidence vault
    if not st.session_state.show_intro:
        # 根据是否需要 evidence 面板决定布局
        if st.session_state.get("show_snippet_panel", False):
            col_chat, col_ctx = st.columns([2.2, 1.8])
        else:
            col_chat, = st.columns([1])
            col_ctx = None

        # ---------- 左侧：聊天 ----------
        with col_chat:
            st.caption("Example question: I’m designing an AR interface to guide warehouse pickers. Main users are novice pickers in large e-commerce warehouses. I mainly care about task completion time and error rate. I’m in the proposal stage with a rough RQ draft. Can you help me improve the research question and check if it fits the IDE proposal rubric?")
            st.caption("Example question: I’m preparing my green-light plan for a mental-health chatbot for university students.Users: Dutch master’s students experiencing study stress.Metrics: engagement (return visits) and perceived support.I have a draft method (diary study + interviews).Can you use the green-light checklist to tell me what is missing?")
            # 展示历史对话
            for i, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # advice 是否已给出
            assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
            advice_given = len(assistant_msgs) > 0

            # 取本轮检索结果（如果有）
            latest_retrieval = st.session_state.retrievals[-1] if st.session_state.retrievals else []

            # 在最后一条 assistant 消息后加“Sources: snippet1, ...”
            if advice_given and latest_retrieval:
                st.markdown("**Sources:**", unsafe_allow_html=True)
                st.caption("Click the snippet to show the raw text at the top. Double click to hide.")
                cols = st.columns(len(latest_retrieval))
                for idx, r in enumerate(latest_retrieval):
                    label = f"snippet{r['rank']}"
                    with cols[idx]:
                        is_selected = (
                                st.session_state.get("selected_snippet_rank") == r["rank"]
                                and st.session_state.get("show_snippet_panel")
                        )

                        clicked = st.button(
                            label,
                            key=f"snippet-btn-{r['rank']}",
                            type=("primary" if is_selected else "secondary"),
                        )

                        if clicked:
                            # 点击之前的状态
                            prev_rank = st.session_state.get("selected_snippet_rank")
                            prev_show = st.session_state.get("show_snippet_panel", False)

                            # 这次点击的结果是 show 还是 hide？
                            if prev_show and prev_rank == r["rank"]:
                                action = "hide"
                            else:
                                action = "show"

                            # 写点击日志
                            user_id = st.session_state.get("user_id") or ""
                            turn_idx = max(0, st.session_state.get("turn_counter", 1) - 1)
                            # turn_counter 在生成答案后 +1，这里通常想绑定到“上一轮回答”
                            log_snippet_click(user_id, turn_idx, r, action)

                            # 更新 UI 状态
                            toggle_snippet_panel(r["rank"])
                            st.rerun()

            # chat input / post-survey 控制
            user_input = None
            if st.session_state.busy:
                st.info("The coach is thinking…")
                st.rerun()
            else:
                if advice_given:
                    st.info("The advice is given for this round.")
                    if not st.session_state.get("post_survey_done", False):
                        if st.button("End this conversation & fill post-survey"):
                            post_survey_dialog()
                    else:
                        st.success("The post-survey is finished, thanks for participating!")
                else:
                    user_input = st.chat_input("ask your questions")

            # 仅在第一次提问时生成回答
            if user_input and not advice_given:
                st.session_state.busy = True
                # 记录问题时间戳
                ts_q = datetime.utcnow().isoformat()
                # 保存用户消息到 session
                st.session_state.messages.append({"role": "user", "content": user_input})

                with st.spinner("Retrieving relevant snippets and generating an answer…"):
                    # 轮次号（先缓存，后面写日志用）
                    turn_idx = st.session_state.turn_counter
                    retrieved = retrieve_top_k(
                        query=user_input,
                        top_k=6,
                    )
                    st.session_state.retrievals.append(retrieved)

                    answer = call_llm_with_context(
                        question=user_input,
                        retrieved_chunks=retrieved,
                    )
                    # 回答时间戳
                    ts_a = datetime.utcnow().isoformat()
                    # -------- 写聊天日志到 CSV --------
                    try:
                        user_id = st.session_state.get("user_id") or ""
                        row = {
                            "user_id": user_id,
                            "turn_index": turn_idx,
                            "timestamp_question": ts_q,
                            "timestamp_answer": ts_a,
                            "question": user_input,
                            "answer": answer,
                            # 把本轮所有 snippet 的关键信息打平成字符串
                            "retrieved_ids": ";".join(str(r.get("id", "")) for r in retrieved),
                            "retrieved_ranks": ";".join(str(r.get("rank", "")) for r in retrieved),
                            "retrieved_scores": ";".join(f"{r.get('score', 0):.4f}" for r in retrieved),
                            "retrieved_source_types": ";".join(str(r.get("source_type", "")) for r in retrieved),
                            "retrieved_doc_titles": ";".join(
                                (r.get("doc_title") or r.get("source_id") or "").replace(";", ",")
                                for r in retrieved
                            ),
                        }
                        append_csv_row(CHAT_LOG_FILE, fieldnames=list(row.keys()), row_dict=row)
                    except Exception as e:
                        # 不要打断用户流程，失败就静默忽略或简单 print
                        print("Failed to log chat turn:", e)

                # 轮次 +1
                st.session_state.turn_counter += 1
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.busy = False
                st.rerun()

        # ---------- 右侧：单条 Retrieved snippet ----------
        if col_ctx is not None:
            with col_ctx:
                st.subheader("Retrieved snippet (this round)")

                latest = st.session_state.retrievals[-1] if st.session_state.retrievals else []
                selected_rank = st.session_state.get("selected_snippet_rank")

                if not latest or selected_rank is None:
                    st.caption("Click a source button below the advice to inspect a snippet.")
                else:
                    # 找到被选中的 snippet
                    snippet = None
                    for r in latest:
                        if r["rank"] == selected_rank:
                            snippet = r
                            break

                    if snippet is None:
                        st.caption("Snippet not found.")
                    else:
                        # st.write("DEBUG meta for", snippet)
                        src = snippet["source_type"]
                        doc = snippet.get("doc_title") or snippet.get("source_id") or "(unknown source)"
                        st.markdown(
                            f"**[{snippet['rank']}] ({src}) {doc}**  \n"
                            f"`similarity score = {snippet['score']:.3f}`"
                        )
                        st.caption(
                            "In the raw text snippet below, you can scroll down in the box to see the whole content.")
                        # 修复 markdown 中的图片链接
                        fixed_md = fix_raw_excerpt_md(snippet["text"], snippet.get("source_id") or "")

                        # 将 markdown 转成 HTML
                        snippet_html = md.markdown(fixed_md)

                        st.markdown(
                            f"""
                    <div style="
                        border: 1px solid #e5e7eb;
                        border-radius: 8px;
                        padding: 0.6rem 0.8rem;
                        height: 30rem;              /* 固定高度 */
                        overflow-y: auto;           /* 竖向滚动 */
                        overflow-x: hidden;
                        width: 100%;                /* 占满右侧列宽度 */
                        box-sizing: border-box;
                    ">
                    {snippet_html}
                    </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # st.markdown("---")
                # st.caption(
                #     "Baseline A = simple vector search over raw markdown chunks "
                #     "(`source_chunk_md`), no Stage × Mode × Gap routing, no tags."
                # )

if __name__ == "__main__":
    main()
