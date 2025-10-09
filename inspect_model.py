#!/usr/bin/env python3
"""
Inspect the PyTorch model - show architecture and parameter counts
"""

import torch
from src.com.saturdaysai.exonet.utils.cnn import ExoplaNET_v1

# Initialize model with same config as training
model = ExoplaNET_v1(
    len_global_lightcurves=2049,
    len_local_lightcurves=257,
    len_secondary_lightcurves=0,
    len_extra_parameters=0
)

# Load trained weights
try:
    model.load_state_dict(torch.load("models/EXOPLANETv1_985.pt", map_location='cpu'))
    print("✓ Loaded trained model weights\n")
except:
    print("⚠ Could not load weights, showing untrained model\n")

print("=" * 80)
print("MODEL ARCHITECTURE: ExoplaNET_v1")
print("=" * 80)

# Count parameters by layer
total_params = 0
trainable_params = 0

print("\n{:<50} {:>15} {:>15}".format("Layer", "Parameters", "Trainable"))
print("-" * 80)

for name, param in model.named_parameters():
    param_count = param.numel()
    total_params += param_count
    if param.requires_grad:
        trainable_params += param_count
    
    trainable_str = "Yes" if param.requires_grad else "No"
    print("{:<50} {:>15,} {:>15}".format(name, param_count, trainable_str))

print("-" * 80)
print("{:<50} {:>15,}".format("TOTAL PARAMETERS", total_params))
print("{:<50} {:>15,}".format("Trainable parameters", trainable_params))
print("{:<50} {:>15,}".format("Non-trainable parameters", total_params - trainable_params))

# Model size in MB
param_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
print("{:<50} {:>15.2f} MB".format("Model size (float32)", param_size_mb))

print("\n" + "=" * 80)
print("MODEL STRUCTURE")
print("=" * 80)
print(model)

print("\n" + "=" * 80)
print("INPUT/OUTPUT SPECS")
print("=" * 80)
print(f"Input shape:  (batch_size, 1, 2306)")
print(f"  - Global view: 2049 values")
print(f"  - Local view:  257 values")
print(f"Output shape: (batch_size, 2)")
print(f"  - Class 0: False Positive")
print(f"  - Class 1: Exoplanet")

# Test forward pass
print("\n" + "=" * 80)
print("FORWARD PASS TEST")
print("=" * 80)
test_input = torch.randn(1, 1, 2306)
model.eval()
with torch.no_grad():
    output = model(test_input)
    probs = torch.softmax(output, dim=1)

print(f"Test input shape:  {test_input.shape}")
print(f"Model output (logits): {output.numpy()}")
print(f"Softmax probabilities: {probs.numpy()}")
print(f"Predicted class: {output.argmax(dim=1).item()}")

print("\n" + "=" * 80)
