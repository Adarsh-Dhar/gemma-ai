from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, pipeline

import torch

model = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model)
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    max_new_tokens=600,
    do_sample=True,
    temperature=1,
    top_k=50,
    top_p=0.95
)
messages = []

app = Flask(__name__)

@app.route("/")
def chat_page():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    chat_message = data.get("chat")
    messages.append({"role": "user", "content": chat_message})
    prompt = text_generation_pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(prompt)
    outputs = text_generation_pipeline.generate(
        prompt,
        max_new_tokens=600,
        do_sample=True,
        temperature=1,
        top_k=50,
        top_p=0.95
    )
    assistant_response = outputs[0]["generated_text"][len(prompt):]

    messages.append({"role": "assistant", "content": assistant_response})

    return jsonify({"chat": assistant_response})

if __name__ == "__main__":
    app.run(debug=True)
