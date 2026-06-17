from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "hi theerreee"}


@app.get("/ask")
def ask_ai(question: str):
    return {"question": question, "answer": "this is mock"}
