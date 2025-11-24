# 📘 **README: Procedural Trust — Thesis Coaching System**

## 🔍 Overview

**ProceduralTrust** is a research system designed to study how procedural features of AI assistants influence user trust, transparency, and perceived fairness.
It includes a **Streamlit-based interactive thesis-coaching application** powered by LLMs and policy-aligned retrieval, along with fully reproducible preprocessing pipelines, corpus indexing, and structured logging for research analysis.

The system supports:

* 📄 **Thesis writing assistance**
* 🔎 **Evidence-based critique** (LLM + policy retrieval)
* 🧭 **Rubric-guided feedback generation**
* 📚 **Policy and thesis corpus indexing**
* 🔬 **Survey-based logging for experiments**
* ⚖ **Baseline (non-procedural) comparison system**

This repository is meant both as an **interactive prototype** and a **research workflow** for studying AI procedural trust.

[Main Coaching Interface](images/main_screen.png)

[Evidence Vault + Rubric-aware Reasoning](images/evidence_card.png)

## 🔲 **Procedural Model: Stage × Mode × Gap**

The Thesis Coach system is built around a **stage × mode × gap** model of procedural AI assistance.

### **1. Stage — Where the student is in the thesis process**

Stages are derived from:

* graduation policy structure
* IDE green-light requirements
* extracted rubrics (`P1–P10`)
* user query context

In the UI, stages appear as collapsible sections (e.g., *Step 1: Administrative requirements*, *Step 2: IDE alignment*, etc.).

---

### **2. Mode — How the system presents procedural information**

Modes include:

* **Rubric mode:** show applicable rubric items (P1–P10)
* **Evidence mode:** retrieve policy excerpts and score relevance
* **Reasoning mode:** explain *why* a requirement applies
* **Action mode:** concrete next steps for the user
* **Transparency mode:** show/hide raw evidence
* **Baseline mode:** opaque, non-procedural alternative

Modes correspond to UI components such as the **Evidence Vault**, rubric tags, inline explanations, and collapsible reasoning blocks.

---

### **3. Gap — What is missing from the student's plan**

Gaps are inferred through:

* rubric criteria
* policy alignment
* LLM comparison against expected requirements

For each step, the system:

1. Identifies missing requirements
2. Explains the reasoning
3. Retrieves supporting evidence
4. Generates actionable tasks

This forms a procedural feedback loop:
**identify gap → justify gap → show evidence → propose action.**


---

## 🧱 Repository Structure

```
ProceduralTrust/
│
├── app.py                      # Main Streamlit application (procedural version)
├── app_baseline.py             # Baseline / non-procedural comparison interface
│
├── baseline_from_indexes_vs/   # Precomputed baseline index (embeddings + entries)
├── policy_docs_index.json      # Main policy index for retrieval
├── thesis_corpus_index.json      # Main thesis index for retrieval
│
├── logs/                       # CSV logs for pre-survey, post-survey, chat turns, evidence events
│
└── .env                        # Local secrets (ignored)
```

---

# 🚀 Application Features

### **1. Interactive Thesis Coaching App (`app.py`)**

A multi-page Streamlit interface enabling:

* Thesis topic input
* Step-by-step coaching
* Policy-based evidence retrieval
* LLM explanations of retrieved text
* Rubric-aligned feedback
* Logging of all interactions (chat turns + evidence events)
* Pre-survey & Post-survey integration

Designed for controlled user studies around transparency and procedural features.

---

### **2. Baseline System (`app_baseline.py`)**

A simplified version of the app that:

* Hides procedural elements
* Removes evidence provenance
* Removes rubrics
* Offers more opaque LLM responses

Used for **between-condition or within-condition comparisons** in trust/fairness experiments.

---

### **3. Corpus Processing Pipeline**

Scripts include:

* **(1) Raw policy extraction**
* **(2) Corpus cleaning + normalization**
* **(3) Embedding generation (OpenAI text-embedding-3-large)**
* **(4) Index creation for retrieval**

These stages support reproducible academic workflows.

---

### **4. Logging System (for research use)**

The application logs:

| Log File                   | Purpose                                     |
| -------------------------- | ------------------------------------------- |
| `pre_survey_ebcs.csv`      | Participant demographic + initial attitudes |
| `post_survey_ebcs.csv`     | Post-interaction trust & fairness measures  |
| `chat_turns_ebcs.csv`      | Every user ⇄ model turn                     |
| `evidence_events_ebcs.csv` | Every time evidence is shown or hidden      |

This enables easy statistical analysis (Python, R, SPSS, JASP, etc.).

---

# 🛠 Installation & Setup

### **1. Clone the repository**

```bash
git clone https://github.com/<your-username>/ProceduralTrust.git
cd ProceduralTrust
```

---

### **2. Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

### **4. Create a `.env` file**

```
OPENAI_API_KEY=sk-xxxx
EBCS_MODEL=gpt-5.1-mini
EBCS_EMBED_MODEL=text-embedding-3-large
```

(Optional) Add custom paths:

```
POLICY_INDEX=policy_docs_index.json
THESES_INDEX=thesis_corpus_index.json
```

---

# ▶ Running the Application

### **Procedural (Full) version**

```bash
streamlit run app.py
```

### **Baseline (Opaque) version**

```bash
streamlit run app_baseline.py
```

Access the app at:

```
http://localhost:8501
```

---

# 🧪 Research Use & Experimental Data Collection

The system was designed to support:

* User studies
* Trust measurement
* Procedural transparency experiments
* Behavioral logging
* Mixed-methods research
* HCI fairness & AI accountability work

Logs can be found in `logs/` and easily exported or downloaded via Streamlit components.

---

# 📦 Preprocessing Pipeline

To rebuild indexes from scratch:

```bash
python 1_Minuer_extraction.py
python 2_preprocess_corpora.py
python 3_RawText_emb.py
```

Or manually clean index files:

```bash
python clean_json.py
```

---

# ☁ Deployment (Lightsail / EC2 / Streamlit Cloud)

The app can be deployed to:

* **Amazon Lightsail** (recommended for your setup)
* **AWS EC2**
* **Streamlit Community Cloud**
* Docker containers (optional)

Full deployment instructions can be generated on request.

---

# 📄 License

(Add your chosen license here — MIT, Apache 2.0, CC BY-NC, etc.)

---

# 🙏 Acknowledgements

This system draws inspiration from research on:

* Procedural justice
* Explainability and transparency
* Human–AI collaboration
* Educational writing support systems
* AI fairness and policy compliance

---

