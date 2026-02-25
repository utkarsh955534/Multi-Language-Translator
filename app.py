from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# HuggingFace Inference API
API_URL = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"
HF_TOKEN = os.environ.get("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")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# language mapping (NLLB codes)
languages = {
    "hindi": "hin_Deva",
    "gujarati": "guj_Gujr",
    "marathi": "mar_Deva",
    "bengali": "ben_Beng",
    "tamil": "tam_Taml",
    "telugu": "tel_Telu",
    "malayalam": "mal_Mlym",
    "punjabi": "pan_Guru",
    "urdu": "urd_Arab",
    "nepali": "npi_Deva",
    "spanish": "spa_Latn",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "italian": "ita_Latn",
    "portuguese": "por_Latn",
    "arabic": "arb_Arab",
    "turkish": "tur_Latn",
    "korean": "kor_Hang",
    "japanese": "jpn_Jpan",
    "chinese": "zho_Hans",
    "thai": "tha_Thai",
    "indonesian": "ind_Latn",
    "vietnamese": "vie_Latn",
    "bhojpuri": "bho_Deva",
    "sanskrit": "san_Deva"
}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


@app.route("/")
def home():
    return render_template("index.html", languages=languages)


@app.route("/translate", methods=["POST"])
def translate_api():
    data = request.get_json()

    text = data.get("text")
    lang = data.get("lang")

    if lang not in languages:
        return jsonify({"translated": "Language not supported"})

    payload = {
        "inputs": text,
        "parameters": {
            "tgt_lang": languages[lang]
        }
    }

    result = query(payload)

    # handle first load / errors
    try:
        translated = result[0]["translation_text"]
    except:
        translated = "Model loading… please try again"

    return jsonify({"translated": translated})


# Render port binding
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
