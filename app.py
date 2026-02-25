from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

app = Flask(__name__)

model_name = "facebook/nllb-200-distilled-600M"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Model loaded on:", device)

# language codes (NLLB)
languages = {
    "hindi": "hin_Deva",
    "gujarati": "guj_Gujr",
    "spanish": "spa_Latn",
    "french": "fra_Latn",
    "russian": "rus_Cyrl",
    "chinese": "zho_Hans",
    "bhojpuri": "bho_Deva",
    "sanskrit": "san_Deva",
    "marathi": "mar_Deva",
    "bengali": "ben_Beng",
    "tamil": "tam_Taml",
    "telugu": "tel_Telu",
    "malayalam": "mal_Mlym",
    "punjabi": "pan_Guru",
    "urdu": "urd_Arab",
    "nepali": "npi_Deva",
    "odia": "ory_Orya",
    "kannada": "kan_Knda",
     "german": "deu_Latn",
    "italian": "ita_Latn",
    "portuguese": "por_Latn",
    "arabic": "arb_Arab",
    "turkish": "tur_Latn",
    "korean": "kor_Hang",
    "japanese": "jpn_Jpan",
    "thai": "tha_Thai",
    "indonesian": "ind_Latn",
    "vietnamese": "vie_Latn"
}


def translate(text, target_lang):

    tokenizer.src_lang = "eng_Latn"   # English source

    inputs = tokenizer(text, return_tensors="pt").to(device)

    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        max_length=200
    )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

@app.route("/")
def home():
    return render_template("index.html", languages=languages)


@app.route("/translate", methods=["POST"])
def translate_api():
    try:
        data = request.get_json()

        text = data.get("text")
        lang = data.get("lang").lower()

        if lang not in languages:
            return jsonify({"translated":"Language not supported"})

        result = translate(text, languages[lang])

        return jsonify({"translated": result})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"translated":"Server error"}),500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
