from flask import Flask, render_template, request
import requests, os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ El token se cargará desde una variable de entorno
HF_TOKEN = os.getenv("HF_TOKEN")

@app.route("/index", methods=["GET", "POST"])
def index():
    result = None
    img_path = None

    if request.method == "POST":
        img = request.files["image"]
        img_path = os.path.join(UPLOAD_FOLDER, img.filename)
        img.save(img_path)

        model_url = "https://api-inference.huggingface.co/models/dima806/facial_emotions_image_detection"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        with open(img_path, "rb") as f:
            response = requests.post(model_url, headers=headers, data=f.read())

        try:
            data = response.json()
            if not data or (isinstance(data, dict) and data.get("error")):
                raise ValueError(data.get("error", "Respuesta vacía de la API"))

            top = sorted(data, key=lambda x: x["score"], reverse=True)[0]
            emotion = top["label"]

            if emotion in ["happy", "surprise"]:
                result = f"Parece que dice la verdad 😄 ({emotion})"
            elif emotion in ["fear", "sad", "angry", "disgust"]:
                result = f"Mmm... parece que oculta algo 🤔 ({emotion})"
            else:
                result = f"Emoción detectada: {emotion}"

        except Exception as e:
            result = f"Error analizando la imagen: {e}"

    return render_template("index.html", result=result, img_path=img_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
