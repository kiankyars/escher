"""
Inspect the model architecture using Modal (where PyTorch is already installed)
"""
import modal

app = modal.App("model-inspector")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.1.0", "numpy==1.24.3")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("models/EXOPLANETv1_985.pt", remote_path="/root/models/EXOPLANETv1_985.pt")
)

@app.function(image=image)
def inspect_model():
    import torch
    import sys
    sys.path.append("/root")
    from src.com.saturdaysai.exonet.utils.cnn import ExoplaNET_v1
    
    # Initialize model
    model = ExoplaNET_v1(
        len_global_lightcurves=2049,
        len_local_lightcurves=257,
        len_secondary_lightcurves=0,
        len_extra_parameters=0
    )
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load("/root/models/EXOPLANETv1_985.pt", map_location='cpu'))
        print("✓ Loaded trained model weights\n")
    except Exception as e:
        print(f"⚠ Could not load weights: {e}\n")
    
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
    param_size_mb = total_params * 4 / (1024 ** 2)
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
    
    print("\n" + "=" * 80)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": param_size_mb
    }

@app.local_entrypoint()
def main():
    result = inspect_model.remote()
    print(f"\n\nSummary: {result['total_params']:,} parameters, {result['model_size_mb']:.2f} MB")
