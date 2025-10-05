# ExoMinerSmall - Minimal Training Setup

## ✅ Setup Complete

All configuration files are ready. You can start training as soon as your TESS dataset is ready.

## 📁 Files Created

```
experiments/local_minimal/
├── model_config_small.yaml   # ExoMinerSmall architecture config
├── config_train.yaml          # Training hyperparameters
├── datasets_fps.yaml          # Dataset paths (update after download)
├── train.sh                   # Training launcher script
├── SETUP.md                   # Detailed setup instructions
└── run_001/                   # Output directory (auto-created)
```

## 🚀 Quick Start

### 1. Install TensorFlow

```bash
# Using conda (recommended)
conda env create -f exominer_pipeline/conda_env_exoplnt_dl_arm64.yml
conda activate exoplnt_dl

# Or using pip
python3 -m pip install tensorflow==2.13.0 numpy pyyaml scikit-learn matplotlib pandas
```

### 2. Wait for Dataset Download

You're currently downloading:
- `tfrecords_tess-spoc-2min_s1-s67_9-24-2024_1159.tar.xz`

This is the **unnormalized** TESS dataset.

### 3. Preprocess the Data

**Important:** The dataset requires preprocessing before training:

#### Step A: Extract
```bash
tar -xJf tfrecords_tess-spoc-2min_s1-s67_9-24-2024_1159.tar.xz
```

#### Step B: Split into train/val/test
```bash
# Use the preprocessing scripts
cd /Users/kian/Code/escher/src_preprocessing/split_tfrecord_train-test
# Follow instructions in that directory
```

#### Step C: Compute normalization statistics
```bash
cd /Users/kian/Code/escher/src_preprocessing/normalize_tfrecord_dataset
# Compute stats from training set
# Apply normalization to all splits
```

See `src_preprocessing/README.md` for detailed instructions.

### 4. Update Dataset Paths

Edit `experiments/local_minimal/datasets_fps.yaml`:

```yaml
train:
  - /path/to/normalized/train/train_shard-0000-of-XXXX
  - /path/to/normalized/train/train_shard-0001-of-XXXX
  # ... more shards

val:
  - /path/to/normalized/val/val_shard-0000-of-XXXX

test:
  - /path/to/normalized/test/test_shard-0000-of-XXXX
```

### 5. Run Training

```bash
cd /Users/kian/Code/escher
conda activate exoplnt_dl  # or your venv
./experiments/local_minimal/train.sh
```

## 🏗️ Architecture: ExoMinerSmall

**Conv Branches:**
- `global_flux`: Full orbit light curve (301 bins) + 1 scalar
- `local_flux`: Transit-focused view (31 bins) + 1 scalar

**Scalar Branch:**
- `stellar`: 3 stellar parameters (Teff, radius, RUWE)

**FC Block:**
- 4 layers, 512 units each
- PReLU activation
- Dropout 0.02

**Estimated:** ~500K-2M parameters (~5-20 MB model)

## 📊 Expected Performance

- **Training time:** 30-60 min on GPU (A10/A100)
- **Per epoch:** ~1-3 min depending on dataset size
- **Early stopping:** Patience 20 epochs on val_auc_pr

## 📈 Monitoring Training

```bash
# Watch live progress
tail -f experiments/local_minimal/run_001/training.log

# View TensorBoard
tensorboard --logdir experiments/local_minimal/run_001/tensorboard
```

## 📦 Output Files

After training:

```
run_001/
├── model.keras              # Trained weights
├── model_summary.txt        # Architecture & param count
├── model.png                # Architecture diagram
├── history.csv              # Metrics per epoch
├── predictions_train.csv    # Training set predictions
├── predictions_val.csv      # Validation set predictions
├── predictions_test.csv     # Test set predictions
├── config_run.yaml          # Full run configuration
└── tensorboard/             # TensorBoard logs
```

## 🔧 Customization

### Reduce model size (faster training)
Edit `model_config_small.yaml`:
```yaml
config:
  num_glob_conv_blocks: 1    # default: 2
  num_loc_conv_blocks: 1     # default: 2
  init_fc_neurons: 256       # default: 512
  num_fc_layers: 2           # default: 4
```

### Adjust training
Edit `config_train.yaml`:
```yaml
training:
  n_epochs: 50               # default: 100
  batch_size: 64             # default: 32 (larger = faster if GPU allows)
callbacks:
  train:
    early_stopping:
      patience: 10           # default: 20
```

## 🐛 Troubleshooting

**"python: command not found"**
→ Your shell script couldn't find python. Activate conda env first or use `python3`

**"ModuleNotFoundError: No module named 'tensorflow'"**
→ Install dependencies (step 1)

**"FileNotFoundError: TFRecord not found"**
→ Update `datasets_fps.yaml` with correct paths

**"InvalidArgumentError: Feature X not found"**
→ Your TFRecords don't match the expected features. Check feature names in your data.

**"OOM (Out of Memory)"**
→ Reduce `batch_size` in `config_train.yaml`

## 🎯 Next Steps for Hackathon

Once training works:

1. **Package for Modal:**
   - Move training script to Modal function
   - Add GPU decorator (@app.function(gpu="A10G"))
   - Volume for datasets

2. **Build inference API:**
   - Load trained model
   - Accept light curve JSON
   - Return exoplanet probability

3. **Integrate with Lovable frontend:**
   - Upload light curve
   - Show live classification
   - Display confidence + explanation

4. **Add real-time learning:**
   - Collect user feedback
   - Scheduled fine-tuning (15-min batches)
   - Shadow deploy + promotion logic

## 📚 References

- ExoMiner paper: https://arxiv.org/abs/2112.04846
- TESS mission: https://tess.mit.edu/
- Original repo: https://github.com/nasa/ExoMiner
