from flask import Flask, request, jsonify

import string
import joblib
import os

# Tentukan path file model relatif terhadap lokasi file aplikasi
model_path = os.path.join(os.path.dirname(__file__), 'models', 'model_bot_sadap.joblib')

# Memuat model
model = joblib.load(model_path)

def preprocess(chat):
    chat = chat.lower()
    tandabaca = tuple(string.punctuation)
    chat = ''.join(ch for ch in chat if ch not in tandabaca)
    return chat

app = Flask(__name__)

@app.route("/")
def home():
    return "<p>Ini adalah API untuk model Machine Learning dari Bot Telegram Sapa Data Empat Lawang (SADAP)</p>"

@app.route("/input")
def handleResponse():
    args = request.args
    text = args.get('text', default='default')
    output = model.predict([preprocess(text)])[0]

    prob = max(model.predict_proba(['ipm'])[0])
    return jsonify({'output': output, 'prob': prob})

# app.run(debug=True)