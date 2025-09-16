# Truss: The Simplest Way to Deploy and Serve Your AI/ML Models

**Truss** empowers you to effortlessly package, deploy, and serve your machine learning models, streamlining the path from development to production.  Learn more and contribute at the [original repo](https://github.com/basetenlabs/truss).

## Key Features of Truss

*   **Write Once, Run Anywhere:**  Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Accelerated Development:**  Benefit from a fast development loop with live reload and a batteries-included model serving environment, eliminating the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:**  Seamlessly deploy models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:**  Easily deploy your models to various platforms, starting with Baseten, and with AWS SageMaker support coming soon.
*   **Built-in Examples:** Explore and deploy popular pre-built models for tasks such as text classification, image generation, and speech recognition.

## Quickstart: Deploy a Text Classification Model

Get started with Truss by packaging a text classification pipeline from the `transformers` package.

### 1. Create a Truss

```bash
truss init text-classification
```

Provide a name, such as "Text classification."

### 2. Navigate to the directory:

```bash
cd text-classification
```

### 3. Implement the Model

Implement the `Model` class in `model/model.py`:

```python
from transformers import pipeline

class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        self._model = pipeline("text-classification")

    def predict(self, model_input):
        return self._model(model_input)
```

### 4. Add Dependencies

Specify dependencies in `config.yaml`:

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

### 5. Deployment

Use [Baseten](https://baseten.co) for deployment.

*   Get a [Baseten API key](https://app.baseten.co/settings/account/api_keys) if you don't already have one.
*   Run `truss push` to deploy your model.

### 6. Invoke the Model

After deployment, invoke the model from the terminal:

```bash
truss predict -d '"Truss is awesome!"'
```

Example Response:

```json
[
  {
    "label": "POSITIVE",
    "score": 0.999873161315918
  }
]
```

## Truss Contributors

Truss is backed by Baseten and built in collaboration with ML engineers worldwide. Special thanks to [Stephan Auerhahn](https://github.com/palp) @ [stability.ai](https://stability.ai/) and [Daniel Sarfati](https://github.com/dsarfati) @ [Salad Technologies](https://salad.com/) for their contributions.

We welcome contributions in accordance with our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).