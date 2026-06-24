from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import cv2
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("./backend/model/model.keras")

with open("./backend/model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("./backend/model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 100

def preprocess_image(file_bytes):
    img_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    img = img.astype(np.float32) / 255.0

    return np.expand_dims(img, axis=0)

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    return padded

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(
    title: str = Form(...),
    description: str = Form(""),
    image: UploadFile = File(...)
):

    text = title + " " + description

    # read image
    img_bytes = await image.read()

    img_input = preprocess_image(img_bytes)
    text_input = preprocess_text(text)

    # prediction
    preds = model.predict({"img": img_input, "text": text_input})
    pred_class = np.argmax(preds, axis=1)[0]

    label = le.inverse_transform([pred_class])[0]
    confidence = float(np.max(preds))

    return {
        "prediction": label,
        "confidence": confidence
    }

frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend/dist")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")