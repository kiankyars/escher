# ExoMinerSmall Training Setup

## Current Status
✅ Configuration files created  
✅ Directory structure ready  
⏳ Waiting for TESS dataset download  
⏳ Python environment setup needed

## Quick Start

### 1. Install Dependencies

You have two options:

#### Option A: Using Conda (Recommended)
```bash
cd /Users/kian/Code/escher
conda env create -f exominer_pipeline/conda_env_exoplnt_dl_arm64.yml
conda activate exoplnt_dl
```

#### Option B: Using pip
```bash
cd /Users/kian/Code/escher
python3 -m venv venv
source venv/bin/activate
pip install tensorflow numpy pyyaml scikit-learn matplotlib pandas
```

### 2. Update Dataset Paths

Once your TESS data download completes:

1. Extract the tar.xz file:
   ```bash
   tar -xJf tfrecords_tess-spoc-2min_s1-s67_9-24-2024_1159.tar.xz
   ```

2. Update `experiments/local_minimal/datasets_fps.yaml` to point to your actual data shards:
   ```yaml
   train:
     - /path/to/tess/data/train/train_shard-0000-of-XXXX
     - /path/to/tess/data/train/train_shard-0001-of-XXXX
     # ... more shards
   
   val:
     - /path/to/tess/data/val/val_shard-0000-of-XXXX
   
   test:
     - /path/to/tess/data/test/test_shard-0000-of-XXXX
   ```

### 3. Run Training

```bash
cd /Users/kian/Code/escher
./experiments/local_minimal/train.sh
```

Or manually:
```bash
python3 src/train/train_model.py \\
    --config experiments/local_minimal/config_train.yaml \\
    --mixed_precision
```

## Architecture: ExoMinerSmall

**What it uses:**
- 2 conv branches: `global_flux` (301 bins) + `local_flux` (31 bins)
- 1 scalar branch: stellar parameters (3 features)
- Default hyperparameters from the original paper

**Estimated size:** ~500K-2M parameters  
**Expected training time:** 30-60 min on GPU (A10/A100)

## Config Files Created

1. **model_config_small.yaml** - Model architecture & hyperparameters
2. **config_train.yaml** - Training settings (epochs, batch size, callbacks)
3. **datasets_fps.yaml** - Dataset file paths (update after download)
4. **train.sh** - Convenient training launcher

## Next Steps

1. ✅ Setup Python environment (conda or pip)
2. ⏳ Wait for TESS dataset download to complete
3. 🔧 Split dataset into train/val/test (see preprocessing docs)
4. 🔧 Compute normalization statistics
5. 🔧 Normalize the TFRecords
6. 🚀 Run training

## Preprocessing Required

The TESS dataset you're downloading is **unnormalized**. You need to:

1. **Split the data:**
   ```bash
   # Use the preprocessing scripts in src_preprocessing/split_tfrecord_train-test/
   ```

2. **Compute normalization stats:**
   ```bash
   # Use src_preprocessing/normalize_tfrecord_dataset/
   ```

3. **Normalize features:**
   ```bash
   # Apply normalization using training set statistics
   ```

See `src_preprocessing/README.md` for detailed instructions.

## Troubleshooting

**"python: command not found"**
→ Use `python3` instead, or activate your conda/venv environment

**"ModuleNotFoundError: No module named 'tensorflow'"**
→ Install dependencies (see step 1)

**"FileNotFoundError: TFRecord not found"**
→ Update `datasets_fps.yaml` with correct paths to your extracted data

## Model Output

After training completes, check `experiments/local_minimal/run_001/`:
- `model.keras` - Trained model weights
- `model_summary.txt` - Architecture details with param count
- `model.png` - Architecture diagram
- `history.csv` - Training metrics per epoch
- `predictions_*.csv` - Model predictions on train/val/test sets
