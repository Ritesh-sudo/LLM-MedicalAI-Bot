from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig

import torch
import requests 
import os

from huggingface_hub import login
login(token="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")


app = Flask(__name__)

# Loading mistral-lora-finetuned-orca model
model_name = "208r1a66g6/mistral-lora-finetuned-orca"
peft_model_name = "208r1a66g6/mistral-lora-finetuned-orca"
config = PeftConfig.from_pretrained(peft_model_name)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
print(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16, device_map="cpu")
print(model.device)
peft_model = PeftModel.from_pretrained(model, peft_model_name)
print(peft_model.device)
chat_pipe = pipeline("text-generation", model=peft_model, tokenizer=tokenizer)
def build_llama_prompt(user_input):
    return (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful and knowledgeable medical assistant. Provide clear, concise, and safe advice.\n"
        "<</SYS>>\n\n"
        f"{user_input.strip()}\n[/INST]"
    )


health_data = {
    "hypertension": "Hypertension is high blood pressure. Maintain a healthy diet, exercise regularly, and follow your doctor's advice.",
    "diabetes": "Diabetes is a chronic condition affecting blood sugar levels. Regular monitoring, medication, and lifestyle changes help manage it.",
    "covid-19": "COVID-19 is caused by the SARS-CoV-2 virus. Follow guidelines from the WHO and local health authorities for prevention and care.",
    "anxiety": "Anxiety is a common mental health condition characterized by excessive worry, nervousness, or fear. It can interfere with daily activities. This information is provided for general awareness and should not be used as a substitute for professional advice."
}

def fetch_health_info(condition):
    """
    Fetch health information from Wikipedia for a given condition.
    If successful, return the page summary (extract); otherwise return a default message.
    """
    condition_for_url = condition.replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{condition_for_url}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "extract" in data:
                return data["extract"]
        return "Information not available."
    except Exception as e:
        return "Information not available."


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/symptom-chatbot', methods=['POST'])
def symptom_chatbot():
    data = request.get_json()
    message = data.get('message', '')
    if not message:
        return jsonify({"error": "No message provided."}), 400

    prompt = build_llama_prompt(message)

    result = chat_pipe(prompt, max_new_tokens=1000, do_sample=True, temperature=0.45)[0]['generated_text'] 
    # using 0.45 temperature for more determinstic output than creative as medicine is based on facts
    print("MODEL RAW OUTPUT:", result)  # Debug print
    response = result.replace(prompt, "").strip()

    return jsonify({"response": response})


@app.route('/health-info', methods=['GET'])
def health_info():
    condition = request.args.get('condition', '').lower()
    if condition in health_data:
        info = health_data[condition]
    else:
        info = fetch_health_info(condition)
        health_data[condition] = info
    return jsonify({"condition": condition, "information": info})


@app.route('/appointment-schedule', methods=['POST'])
def appointment_schedule():
    data = request.get_json()
    name = data.get('name', '')
    preferred_date = data.get('preferred_date', '')
    visit_reason = data.get('visit_reason', '')
    if not name or not preferred_date or not visit_reason:
        return jsonify({"error": "Missing appointment details. Provide name, preferred_date, and visit_reason."}), 400
    
    # Generate a simple confirmation ID.
    confirmation_id = "APPT" + str(abs(hash(name + preferred_date + visit_reason)))[0:6]
    return jsonify({
        "message": "Appointment scheduled successfully.",
        "confirmation_id": confirmation_id,
        "details": {
            "name": name,
            "preferred_date": preferred_date,
            "visit_reason": visit_reason
        }
    })


if __name__ == '__main__':
    app.run(debug=True)