from flask import Flask, render_template, request, jsonify
import requests
import os
import time

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"
HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

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


# ⭐ auto retry while model wakes
def query_with_retry(payload, retries=5):
    for _ in range(retries):
        result = query(payload)

        # model waking
        if isinstance(result, dict) and "estimated_time" in result:
            time.sleep(5)
            continue

        return result

    return {"error": "Model still loading"}


@app.route("/")
def home():
    return render_template("index.html", languages=languages)


@app.route("/translate", methods=["POST"])
def translate_api():
    data = request.get_json()

    text = data.get("text")
    lang = data.get("lang")

    if not text:
        return jsonify({"translated": "Enter some text"})

    if lang not in languages:
        return jsonify({"translated": "Language not supported"})

    payload = {
        "inputs": text,
        "parameters": {
            "tgt_lang": languages[lang]
        }
    }

    result = query_with_retry(payload)

    try:
        translated = result[0]["translation_text"]
    except:
        translated = "Model is busy… try again shortly"

    return jsonify({"translated": translated})


# Render port binding
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
