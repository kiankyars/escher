# 🚀 Your API is LIVE on Modal!

## Production Endpoint

```
https://kiankyars--exoplanet-detector-fastapi-app.modal.run
```

## Quick Test

```bash
curl https://kiankyars--exoplanet-detector-fastapi-app.modal.run/
```

## Lovable Integration

### API Endpoint
```javascript
const API_URL = "https://kiankyars--exoplanet-detector-fastapi-app.modal.run/predict";
```

### Request Format
```javascript
const response = await fetch(API_URL, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    global_view: [/* 2049 float values */],
    local_view: [/* 257 float values */]
  })
});

const result = await response.json();
```

### Response Format
```json
{
  "is_exoplanet": true,
  "confidence": 0.95,
  "probability_exoplanet": 0.95,
  "probability_false_positive": 0.05
}
```

## Example Usage

```javascript
// Example with random data for testing
const testData = {
  global_view: Array(2049).fill(0).map(() => Math.random() * 2 - 1),
  local_view: Array(257).fill(0).map(() => Math.random() * 2 - 1)
};

fetch('https://kiankyars--exoplanet-detector-fastapi-app.modal.run/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(testData)
})
.then(r => r.json())
.then(data => console.log('Prediction:', data));
```

## Dashboard

View logs and metrics:
https://modal.com/apps/kiankyars/main/deployed/exoplanet-detector

## Redeploy

Any time you update the model or code:
```bash
uv run modal deploy modal_deploy.py
```
