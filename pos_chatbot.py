from flask import Flask, request, jsonify
import os
import json
from transformers import pipeline

app = Flask(__name__)

# প্রশ্ন-উত্তর ডাটাবেস লোড করা
FAQ_FILE = "faq.json"
def load_faq():
    try:
        with open(FAQ_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"questions": []}

db = load_faq()

# AI মডেল সেটআপ
qa_model = pipeline("text2text-generation", model="google/flan-t5-small")

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get("query", "").strip()
    
    if not user_input:
        return jsonify({"answer": "অনুগ্রহ করে একটি প্রশ্ন লিখুন।"})
    
    # FAQ থেকে উত্তর খোঁজা
    for qa in db["questions"]:
        if user_input.lower() in qa["query"].lower():
            return jsonify({"answer": qa["answer"]})
    
    # AI মডেল দিয়ে উত্তর তৈরি করা
    generated_answer = qa_model(user_input, max_length=50, num_return_sequences=1)[0]['generated_text']
    
    # নতুন প্রশ্ন ও উত্তর সংরক্ষণ করা
    db["questions"].append({"query": user_input, "answer": generated_answer})
    with open(FAQ_FILE, "w", encoding="utf-8") as file:
        json.dump(db, file, indent=4, ensure_ascii=False)
    
    return jsonify({"answer": generated_answer})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render 5000 পোর্ট ব্যবহার করবে
    app.run(host='0.0.0.0', port=port, debug=True)
