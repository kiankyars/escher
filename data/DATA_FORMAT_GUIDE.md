# 📊 Data Format Guide: What to Upload

## TL;DR - The Quick Answer

Your model expects **2306 normalized flux values** from a **folded light curve**:
- **2049 values**: Global view (full orbital period, folded)
- **257 values**: Local view (zoomed into transit region)

## What is Light Curve Data?

A **light curve** is a graph showing how a star's brightness changes over time. When a planet passes in front of a star (transits), the brightness dips slightly. The model detects these patterns.

### Example Light Curve Structure

```
Time (days)    Flux (normalized)
0.0            -0.025
0.1            -0.023
0.2            -0.120  ← Transit dip!
0.3            -0.115  ← Transit continues
0.4            -0.022
...
```

## The Model's Input Format

### What "Folded" Means

The preprocessing code **folds** the light curve by the orbital period:
1. Take raw time-series data
2. Divide by orbital period
3. Stack all periods on top of each other
4. Creates one "average" transit

**Why?** Makes the transit signal clearer and reduces noise.

### Global View (2049 values)
- **Phase range**: -0.5 to +0.5 (one full orbital period)
- **What it shows**: Complete folded light curve
- **Purpose**: Context - is this a clean signal?

### Local View (257 values)  
- **Phase range**: Zoomed into the transit (typically ±4× transit duration)
- **What it shows**: Just the transit region
- **Purpose**: Detailed transit shape for classification

### Normalization Formula

```python
normalized_flux = (flux - global_median) / (global_median - local_min)
```

This transforms flux to roughly the range **[-1, 0]**.

## How to Get Data to Upload

### Option 1: Use Existing Preprocessed Data ✅ EASIEST

If you have access to the original `.npy` files from the project:

```python
import numpy as np

# Load preprocessed data
X = np.load('X_fold_noCentroid_noNans.npy')

# Each row is one sample ready to upload
sample = X[0]  # Shape: (2306,)

# Save as CSV
np.savetxt('sample_lightcurve.csv', sample, delimiter=',')
```

Upload the CSV directly to your app!

### Option 2: Download and Process from Kepler Archive

#### Step 1: Get Kepler Metadata
Download from NASA's Kepler mission data:
- Archive: https://archive.stsci.edu/
- You need: **KOI ID, Period, T0 (epoch), Transit Duration**

Example from your `features_train.csv`:
```
KOI: K00752.01 (Kepler-227 b)
Period: 9.488036 days
T0: 170.538750
Duration: 0.123229 days (2.96 hours)
```

#### Step 2: Process with LightKurve

```python
import lightkurve as lk
import numpy as np

# Download raw light curve
koi_id = 10797460  # KIC ID for Kepler-227
lc_raw = lk.search_lightcurvefile(f'KIC {koi_id}').download_all().PDCSAP_FLUX.stitch()

# Clean and flatten
lc_flat = lc_raw.flatten().remove_outliers(sigma=20, sigma_upper=4)

# Fold by the orbital period
t0 = 170.538750
period = 9.488036
lc_folded = lc_flat.fold(period, t0)

# Bin to 2049 values (global view)
lc_global_binned = lc_folded.bin(bins=2049, method='median')

# Zoom to transit and bin to 257 values (local view)
duration = 0.123229
local_width = 4 * (duration / period)  # ±4× transit duration
lc_local = lc_folded[(lc_folded.phase > -local_width/2) & (lc_folded.phase < local_width/2)]
lc_local_binned = lc_local.bin(bins=257, method='median')

# Normalize
global_median = np.nanmedian(lc_global_binned.flux)
local_min = np.nanmin(lc_local_binned.flux)
normalize = lambda f: (f - global_median) / (global_median - local_min)

global_normalized = normalize(lc_global_binned.flux)
local_normalized = normalize(lc_local_binned.flux)

# Combine into 2306 values
final_data = np.concatenate([global_normalized, local_normalized])

# Save as CSV
np.savetxt('kepler227b_lightcurve.csv', final_data, delimiter=',')
```

### Option 3: Generate Test Data (for debugging)

```python
import numpy as np

# Generate synthetic light curve with a transit-like dip
def generate_test_lightcurve():
    # Global view: mostly flat with a dip
    global_view = np.random.normal(-0.02, 0.01, 2049)
    global_view[1000:1050] -= 0.1  # Add a transit dip
    
    # Local view: zoomed into the dip
    local_view = np.random.normal(-0.02, 0.01, 257)
    local_view[100:157] -= 0.12  # Deeper dip
    
    return np.concatenate([global_view, local_view])

test_data = generate_test_lightcurve()
np.savetxt('test_lightcurve.csv', test_data, delimiter=',')
```

## File Format Examples

### CSV Format (Recommended)
```csv
-0.0234,0.0123,-0.0456,0.0234,...  (2049 values)
-0.0345,0.0123,-0.0678,0.0234,...  (257 values)
```

### JSON Format
```json
{
  "global_view": [-0.0234, 0.0123, -0.0456, ...],  // 2049 values
  "local_view": [-0.0345, 0.0123, -0.0678, ...]    // 257 values
}
```

### Single Array JSON
```json
[-0.0234, 0.0123, ..., -0.0345, 0.0123, ...]  // 2306 values total
```

## Real Kepler Data Sources

### MAST Archive
- URL: https://archive.stsci.edu/
- Search for: "Kepler Objects of Interest (KOI)"
- Download: Light curve FITS files

### Lightkurve Python Library
```bash
pip install lightkurve
```

```python
import lightkurve as lk

# Search and download
lk.search_lightcurvefile('Kepler-90').download()
```

### NASA Exoplanet Archive
- URL: https://exoplanetarchive.ipac.caltech.edu/
- Has confirmed planets with full metadata

## Quick Test Workflow

1. **Generate test data:**
   ```bash
   python -c "import numpy as np; np.savetxt('test.csv', np.random.randn(2306), delimiter=',')"
   ```

2. **Upload to your app** and verify it works

3. **Get real Kepler data** using one of the methods above

4. **Process it** to match the 2306-value format

5. **Upload and see real predictions!**

## Debugging Tips

### "Insufficient data points"
- Count values: Should be exactly 2306
- Check for headers or comments in CSV

### "Invalid values"
- Ensure all values are numbers
- Remove NaN or infinite values
- Values typically in range [-1, 1]

### "Poor predictions"
- Data might not be normalized correctly
- Check that transit is visible in the data
- Ensure proper folding by orbital period

## Example: Using Data from Your Project

Your project has preprocessed data in `/data/` folder:

```bash
cd /Users/kian/Code/escher

# Extract one sample from the numpy files
python << EOF
import numpy as np
X = np.load('data/X_fold_noCentroid_noNans.npy')
# Save first sample (known planet)
np.savetxt('sample_planet.csv', X[0], delimiter=',')
# Save a false positive
np.savetxt('sample_false_positive.csv', X[100], delimiter=',')
EOF

# Now upload sample_planet.csv to your Lovable app!
```

## Summary

**What you need:** 2306 normalized flux values representing a folded light curve

**Where to get it:**
1. ✅ Use existing `.npy` files from the project
2. 📥 Download from MAST/Kepler archive + process with lightkurve  
3. 🧪 Generate synthetic test data

**Format:** CSV with comma-separated values, or JSON with explicit structure
