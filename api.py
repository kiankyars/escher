from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import softmax
from src.com.saturdaysai.exonet.utils.cnn import ExoplaNET_v1

app = FastAPI(title="Exoplanet Detection API")

# CORS for Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ExoplaNET_v1(
    len_global_lightcurves=2049,
    len_local_lightcurves=257,
    len_secondary_lightcurves=0,
    len_extra_parameters=0
).to(device)

model.load_state_dict(torch.load("models/EXOPLANETv1_985.pt", map_location=device))
model.eval()

class LightCurveInput(BaseModel):
    global_view: list[float]  # 2049 values
    local_view: list[float]   # 257 values

class PredictionOutput(BaseModel):
    is_exoplanet: bool
    confidence: float
    probability_exoplanet: float
    probability_false_positive: float

@app.get("/")
def health_check():
    return {"status": "healthy", "model": "ExoplaNET_v1", "device": str(device)}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: LightCurveInput):
    try:
        # Validate input lengths
        if len(data.global_view) != 2049:
            raise HTTPException(400, f"global_view must be 2049 values, got {len(data.global_view)}")
        if len(data.local_view) != 257:
            raise HTTPException(400, f"local_view must be 257 values, got {len(data.local_view)}")
        
        # Prepare input tensor
        light_curve = np.concatenate([data.global_view, data.local_view])
        x = torch.from_numpy(light_curve).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            logits = model(x)
            probs = softmax(logits.cpu().numpy()[0])
        
        prob_false_positive = float(probs[0])
        prob_exoplanet = float(probs[1])
        
        return PredictionOutput(
            is_exoplanet=prob_exoplanet > 0.5,
            confidence=max(prob_exoplanet, prob_false_positive),
            probability_exoplanet=prob_exoplanet,
            probability_false_positive=prob_false_positive
        )
    
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
