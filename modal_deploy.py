"""Modal deployment - cloud hosting for your model"""
import modal

# Create Modal app
app = modal.App("exoplanet-detector")

# Define container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.104.1",
        "torch==2.1.0",
        "numpy==1.24.3",
        "scipy==1.11.3",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("models/EXOPLANETv1_985.pt", remote_path="/root/models/EXOPLANETv1_985.pt")
)

@app.function(
    image=image,
    cpu=2,  # Use 2 CPUs
    memory=4096,  # 4GB RAM
    min_containers=1,  # Keep 1 instance warm for fast responses
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import torch
    import torch.nn.functional as F
    import numpy as np
    from scipy.special import softmax
    import sys
    
    sys.path.append("/root")
    from src.com.saturdaysai.exonet.utils.cnn import ExoplaNET_v1

    web_app = FastAPI(title="Exoplanet Detection API")

    # CORS
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load model
    device = torch.device("cpu")  # Modal uses CPU by default
    model = ExoplaNET_v1(
        len_global_lightcurves=2049,
        len_local_lightcurves=257,
        len_secondary_lightcurves=0,
        len_extra_parameters=0,
    ).to(device)
    
    model.load_state_dict(torch.load("/root/models/EXOPLANETv1_985.pt", map_location=device))
    model.eval()

    class LightCurveInput(BaseModel):
        global_view: list[float]
        local_view: list[float]

    class PredictionOutput(BaseModel):
        is_exoplanet: bool
        confidence: float
        probability_exoplanet: float
        probability_false_positive: float

    @web_app.get("/")
    def health():
        return {"status": "healthy", "model": "ExoplaNET_v1"}

    @web_app.post("/predict", response_model=PredictionOutput)
    def predict(data: LightCurveInput):
        try:
            if len(data.global_view) != 2049:
                raise HTTPException(400, f"global_view must be 2049 values")
            if len(data.local_view) != 257:
                raise HTTPException(400, f"local_view must be 257 values")
            
            light_curve = np.concatenate([data.global_view, data.local_view])
            x = torch.from_numpy(light_curve).float().unsqueeze(0).unsqueeze(0).to(device)
            
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
    
    return web_app
