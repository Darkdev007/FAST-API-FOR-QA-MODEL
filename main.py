import torch
from fastapi import FastAPI, Depends, HTTPException, Header
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer,AutoModelForQuestionAnswering

load_dotenv()

API_KEY_CREDITS = {os.getenv("API_KEY") : 5}
app = FastAPI()


model_checkpoint = "Darkdev007/bert-finetuned-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def verify_api_key(x_api_key: str = Header(None)):
    credits = API_KEY_CREDITS.get(x_api_key,0)
    if credits <= 0:
        raise HTTPException(status_code=401, detail="Invalid API key or no Credits")

    return x_api_key

@app.post("/generate")
def generate(context:str, question:str, x_api_key : str = Depends(verify_api_key)):
    API_KEY_CREDITS[x_api_key] -=1
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.decode(inputs["input_ids"][0][start:end])
    return {"answer" : answer}