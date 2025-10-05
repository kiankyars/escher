# Quick Start - Local API

## 1. Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Run the API
```bash
python api.py
```

API will run on `http://localhost:8000`

## 3. Test it
```bash
python test_api.py
```

## 4. Connect your Lovable app
Point your frontend to: `http://localhost:8000/predict`

**Request format:**
```json
{
  "global_view": [2049 float values],
  "local_view": [257 float values]
}
```

**Response format:**
```json
{
  "is_exoplanet": true,
  "confidence": 0.95,
  "probability_exoplanet": 0.95,
  "probability_false_positive": 0.05
}
```

---

# Deploy to Modal (Optional)

If you want cloud hosting instead of local:

```bash
pip install modal
modal setup  # First time only
modal deploy modal_deploy.py
```

Modal will give you a public URL.
