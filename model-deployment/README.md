# Deploying models using Flask

### Usage

```bash
python run.py

curl -X POST http://localhost:5001/predict  -H "Content-Type: application/json" -d @input.json
```