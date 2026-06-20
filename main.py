from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
import os
import requests
import base64

load_dotenv()

app = FastAPI()

API_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Add it to your .env file.")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Convert audio to base64 so we can send parameters with it
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "inputs": audio_base64,
        "parameters": {
            "generate_kwargs": {
                "language": "fa",
                "task": "transcribe"
            }
        }
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text
        )

    result = response.json()

    return {
        "language": "fa",
        "text": result.get("text", result)
    }
