"""Quick test script for the API"""
import requests
import numpy as np

# Generate dummy light curve data
global_view = np.random.randn(2049).tolist()
local_view = np.random.randn(257).tolist()

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "global_view": global_view,
        "local_view": local_view
    }
)

print("Status:", response.status_code)
print("Response:", response.json())
