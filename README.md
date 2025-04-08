# LLM-MedicalAI-Bot

A conversational AI system designed to assist users with medical symptom triage, health information, and appointment scheduling. Built using a fine-tuned **Mistral 7B** language model with **LoRA** adapters and deployed via a Flask web server.

---

## 💡 Features

- 🧠 **Chatbot with Symptom Triage:**
  Responds to user symptoms using a medical-tuned LLM.

- 🌐 **Health Info Fetching:**
  Dynamically pulls condition summaries from Wikipedia if not predefined.

- 📅 **Appointment Scheduling:**
  Basic scheduling interface for name, date, and reason via REST API.

- 🧪 **Fine-Tuned Mistral LLM:**
  Uses PEFT + LoRA for efficient fine-tuning on medical dialogue.

- 💻 **Runs on Apple M1/M2 (MPS):**
  Optimized for local use with Hugging Face + Transformers.

---

## 🧰 Tech Stack

- **Python**
- **Flask** (Backend + Web UI)
- **Hugging Face Transformers**
- **PEFT (LoRA)**
- **Mistral 7B** (Base Model)
- **Wikipedia REST API**

---
## 🚀 Getting Started

### 1. Clone this repository
```bash
git clone https://github.com/your-username/medical-ai-chatbot.git
cd medical-ai-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask App
```bash
python Agent.py
```
Then open `http://127.0.0.1:5000` in your browser.

---
