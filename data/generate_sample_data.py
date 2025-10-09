#!/usr/bin/env python3
"""
Generate sample light curve data files for testing the exoplanet detection model.
Creates realistic-looking folded light curves with transit signatures.
Pure Python - no dependencies required.
"""

import json
import math
import random

def gaussian(x, amplitude, center, width):
    """Gaussian function"""
    return amplitude * math.exp(-(x - center)**2 / (2 * width**2))

def generate_global_view(has_planet=True, noise_level=0.001):
    """Generate 2049 values for global folded light curve"""
    flux = []
    for i in range(2049):
        phase = -0.5 + (i / 2048.0)
        
        # Base flux normalized around 0
        value = 0.0
        
        # Add low frequency stellar variability
        value += 0.002 * math.sin(2 * math.pi * phase * 2)
        
        if has_planet:
            # Add transit dip at phase 0
            value -= gaussian(phase, 0.015, 0, 0.017)
        
        # Add random noise
        value += random.gauss(0, noise_level)
        
        flux.append(value)
    
    return flux

def generate_local_view(has_planet=True, noise_level=0.0005):
    """Generate 257 values for local zoomed view around transit"""
    flux = []
    for i in range(257):
        phase = -0.08 + (i / 256.0) * 0.16
        
        # Base flux
        value = 0.0
        
        if has_planet:
            # Transit signal - more pronounced
            value -= gaussian(phase, 0.018, 0, 0.013)
        
        # Add noise
        value += random.gauss(0, noise_level)
        
        flux.append(value)
    
    return flux

def create_sample(has_planet=True, noise_global=0.001, noise_local=0.0005):
    """Create complete sample (2306 values)"""
    global_view = generate_global_view(has_planet, noise_global)
    local_view = generate_local_view(has_planet, noise_local)
    return global_view + local_view

def save_csv(filename, data, description):
    """Save data as CSV"""
    with open(filename, 'w') as f:
        f.write(f'# {description}\n')
        f.write('# 2306 values: 2049 global view + 257 local view\n')
        # Write 10 values per line for readability
        for i in range(0, len(data), 10):
            line = ','.join(f'{v:.8f}' for v in data[i:i+10])
            f.write(line)
            if i + 10 < len(data):
                f.write(',\n')
            else:
                f.write('\n')

def save_json(filename, data, description):
    """Save data as JSON"""
    with open(filename, 'w') as f:
        json.dump({
            'global_view': data[:2049],
            'local_view': data[2049:],
            'description': description
        }, f, indent=2)

# Set random seed for reproducibility
random.seed(42)

print("Generating sample data files...")

# 1. Confirmed exoplanet (strong signal, low noise)
confirmed = create_sample(has_planet=True, noise_global=0.0005, noise_local=0.0003)
save_csv('/Users/kian/Code/escher-d5f2232d/public/sample_confirmed_exoplanet.csv',
         confirmed, 'Strong transit signal - CONFIRMED exoplanet')
save_json('/Users/kian/Code/escher-d5f2232d/public/sample_confirmed_exoplanet.json',
          confirmed, 'Strong transit signal - CONFIRMED exoplanet')

# 2. Candidate exoplanet (weaker signal, more noise)
random.seed(123)
candidate = create_sample(has_planet=True, noise_global=0.002, noise_local=0.001)
save_csv('/Users/kian/Code/escher-d5f2232d/public/sample_candidate_exoplanet.csv',
         candidate, 'Weak transit signal - CANDIDATE exoplanet')
save_json('/Users/kian/Code/escher-d5f2232d/public/sample_candidate_exoplanet.json',
          candidate, 'Weak transit signal - CANDIDATE exoplanet')

# 3. False positive (no planet, just noise)
random.seed(456)
false_positive = create_sample(has_planet=False, noise_global=0.0015, noise_local=0.0008)
save_csv('/Users/kian/Code/escher-d5f2232d/public/sample_false_positive.csv',
         false_positive, 'No transit signal - FALSE POSITIVE')
save_json('/Users/kian/Code/escher-d5f2232d/public/sample_false_positive.json',
          false_positive, 'No transit signal - FALSE POSITIVE')

print("✓ Generated 6 sample files in public/:")
print("  CSV:")
print("    sample_confirmed_exoplanet.csv")
print("    sample_candidate_exoplanet.csv")
print("    sample_false_positive.csv")
print("  JSON:")
print("    sample_confirmed_exoplanet.json")
print("    sample_candidate_exoplanet.json")
print("    sample_false_positive.json")
print("\nUpload these to test your model!")