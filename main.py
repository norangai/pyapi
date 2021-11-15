from fastapi import FastAPI
import numpy as np
import uvicorn
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from pydantic import BaseModel
#import torch
 
app = FastAPI()
 
class Data(BaseModel):
    sentence: str
 
@app.post("/sentiment")
async def sentiment_classification(data:Data):
   
    model.eval()
    pt_inputs = tokenizer(data.sentence, return_tensors="pt")
    #with torch.no_grad():
    output = model(**pt_inputs)
   
    return {"message": labels[np.argmax(output.logits.cpu().numpy())]}
 
if __name__ == "__main__":
    tokenizer = ElectraTokenizer.from_pretrained('hfl/chinese-electra-180g-small-discriminator')
    model = ElectraForSequenceClassification.from_pretrained("senti_clf_v1")
    labels = ['cry', 'ok', 'smile']
    uvicorn.run(app, host="0.0.0.0", port=8000,reload=True)