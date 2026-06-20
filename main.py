from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

API_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    if HF_TOKEN is None:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set")

    audio_bytes = await file.read()

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": file.content_type or "application/octet-stream"
        },
        data=audio_bytes,
        timeout=120
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)

    result = response.json()

    return result
