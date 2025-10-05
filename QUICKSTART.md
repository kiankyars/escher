# ExoMiner Training - Quick Start Guide

## ✅ What's Been Set Up

Complete training configuration for **ExoMinerSmall** architecture:

```
experiments/local_minimal/
├── model_config_small.yaml   # Architecture: 2 conv branches + 1 scalar
├── config_train.yaml          # Training: 100 epochs, batch=32, early stopping
├── datasets_fps.yaml          # Dataset paths (UPDATE after preprocessing)
├── train.sh                   # One-command training launcher
├── README.md                  # Detailed guide
└── SETUP.md                   # Step-by-step instructions
```

## 🎯 Current Status

✅ Config files ready  
✅ Training script ready  
⏳ **WAITING:** TESS dataset download + preprocessing  
⏳ **TODO:** Install TensorFlow

## 🚀 Next Steps (In Order)

### 1. Install Dependencies (5 min)

```bash
cd /Users/kian/Code/escher
conda env create -f exominer_pipeline/conda_env_exoplnt_dl_arm64.yml
conda activate exoplnt_dl
```

Or with pip:
```bash
python3 -m pip install tensorflow==2.13.0 numpy pyyaml scikit-learn matplotlib pandas
```

### 2. Wait for Dataset Download

You're downloading: `tfrecords_tess-spoc-2min_s1-s67_9-24-2024_1159.tar.xz`

### 3. Preprocess Data (30-60 min)

**Important:** The dataset is unnormalized. You MUST:

1. **Extract:**
   ```bash
   tar -xJf tfrecords_tess-spoc-2min_s1-s67_9-24-2024_1159.tar.xz
   ```

2. **Split into train/val/test:**
   ```bash
   # See: src_preprocessing/split_tfrecord_train-test/README.md
   ```

3. **Normalize:**
   ```bash
   # See: src_preprocessing/normalize_tfrecord_dataset/README.md
   ```

### 4. Update Dataset Paths

Edit `experiments/local_minimal/datasets_fps.yaml`:
```yaml
train:
  - /absolute/path/to/normalized/train/train_shard-0000-of-XXXX
  # ... more shards
val:
  - /absolute/path/to/normalized/val/val_shard-0000-of-XXXX
test:
  - /absolute/path/to/normalized/test/test_shard-0000-of-XXXX
```

### 5. Run Training! 🎉

```bash
conda activate exoplnt_dl
cd /Users/kian/Code/escher
./experiments/local_minimal/train.sh
```

## 📊 What to Expect

- **Model:** ExoMinerSmall (~500K-2M params)
- **Training time:** 30-60 min on GPU (A10/A100)
- **Per epoch:** 1-3 min
- **Auto-stop:** Early stopping after 20 epochs without improvement

## 🔍 Monitor Progress

```bash
# Live log
tail -f experiments/local_minimal/run_001/training.log

# TensorBoard
tensorboard --logdir experiments/local_minimal/run_001/tensorboard
```

## 📦 Output

After training:
- `run_001/model.keras` - Trained weights
- `run_001/model_summary.txt` - Param count
- `run_001/predictions_*.csv` - All predictions
- `run_001/history.csv` - Training curves

## 🎯 For the Hackathon

Once you have a trained model:

1. **Test inference locally:**
   ```python
   from tensorflow import keras
   model = keras.models.load_model('experiments/local_minimal/run_001/model.keras')
   # Load test light curve and predict
   ```

2. **Move to Modal for deployment:**
   - Package model + inference code
   - Create GPU-backed inference endpoint
   - Connect to Lovable frontend

3. **Add real-time learning:**
   - Collect user feedback
   - Schedule periodic fine-tuning
   - Deploy with validation gates

## 🐛 Quick Fixes

**"python: command not found"** → Activate conda env or use `python3`  
**"TensorFlow not found"** → Install deps (step 1)  
**"TFRecord not found"** → Update paths in `datasets_fps.yaml`  
**"Feature not found"** → Check your TFRecords have required features  
**OOM error** → Reduce `batch_size` in `config_train.yaml`

## 📚 Documentation

- Full guide: `experiments/local_minimal/README.md`
- Setup steps: `experiments/local_minimal/SETUP.md`
- Preprocessing: `src_preprocessing/README.md`
- Model code: `models/models_keras.py` (line 317: ExoMinerSmall)

## ⚡ Speed Tips

To train faster (with slight performance hit):

Edit `model_config_small.yaml`:
```yaml
config:
  num_glob_conv_blocks: 1    # halve global branch
  num_loc_conv_blocks: 1     # halve local branch
  init_fc_neurons: 256       # smaller FC layers
  num_fc_layers: 2           # fewer FC layers
```

This gives ~200K params and 10-20 min training time.

---

**Questions?** Check the detailed guides or the ExoMiner paper: https://arxiv.org/abs/2112.04846
