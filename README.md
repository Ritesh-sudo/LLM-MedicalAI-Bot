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
