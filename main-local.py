from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
import tempfile
import os
import torch

app = FastAPI(title="Persian Whisper Local API")

MODEL_PATH = r"C:\Users\Sahar\.cache\huggingface\hub\models--Paulwalker4884--whisper-persian\snapshots\80f96e52051b23f239db9b9798c41ed04d5aa568"

DEVICE = 0 if torch.cuda.is_available() else -1
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model once at startup
transcriber = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_PATH,
    device=DEVICE,
    torch_dtype=TORCH_DTYPE,
    chunk_length_s=30,
)


@app.get("/")
def root():
    return {
        "status": "running",
        "model": "openai/whisper-large-v3",
        "format": "flac",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Accept only FLAC extension
    if not file.filename.lower().endswith(".flac"):
        raise HTTPException(
            status_code=400,
            detail="Only FLAC format is supported.",
        )

    # Some clients send different content-types for FLAC
    allowed_content_types = [
        "audio/flac",
        "audio/x-flac",
        "application/octet-stream",
    ]

    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}. Only FLAC allowed.",
        )

    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    temp_path = None

    try:
        # Save FLAC temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        result = transcriber(
            temp_path,
            generate_kwargs={
                "language": "fa",
                "task": "transcribe",
                "temperature": 0,
            },
        )

        return {
            "language": "fa",
            "text": result["text"].strip(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
