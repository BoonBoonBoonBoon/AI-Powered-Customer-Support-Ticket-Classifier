import os
import json
from app.models.runtime_loader import ensure_model_ready
from app.models.inference_transformer import TransformerInference

def main():
    loaded = ensure_model_ready(os.getenv('SERVE_MODEL_ENV', 'staging'))
    manifest = loaded['manifest']
    artifacts = {k: str(v) for k, v in loaded['artifacts'].items()}
    rt = TransformerInference(manifest, artifacts)
    sample = {
        'title': 'Server outage',
        'description': 'Production API returning 500 for all requests'
    }
    out = rt.predict(sample['title'], sample['description'])
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
