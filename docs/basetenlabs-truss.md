# Truss: The Simplest Way to Deploy Your AI/ML Models

**Truss** is a powerful tool that simplifies the process of deploying and serving your machine learning models in production.  [View the original repository](https://github.com/basetenlabs/truss).

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features of Truss

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Fast Developer Loop:** Enjoy rapid iteration with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:**  Supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Easily deploy models to Baseten (with AWS SageMaker support coming soon).
*   **Pre-built Examples:**  Utilize pre-packaged Trusses for popular models like Llama 2, Stable Diffusion XL, and Whisper, and explore [dozens more examples](https://github.com/basetenlabs/truss-examples/).

## Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

## Quickstart: Deploy a Text Classification Model

Here's how to quickly package and deploy a text classification model using Truss:

### 1. Create a Truss

```bash
truss init text-classification
```

Enter a name like `Text classification` when prompted.

```bash
cd text-classification
```

### 2. Implement the Model (`model/model.py`)

Create a `Model` class within the `model/model.py` file:

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

### 3. Add Model Dependencies (`config.yaml`)

Update `config.yaml` to include the required dependencies:

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

## Deployment with Baseten

### 1. Get a Baseten API Key

Obtain your [Baseten API key](https://app.baseten.co/settings/account/api_keys) from your Baseten account.  If you don't have one, [sign up for a free account](https://app.baseten.co/signup/).

### 2. Deploy Your Model

```bash
truss push
```

Follow the prompts to deploy your model to Baseten.  Monitor your deployment via [your model dashboard on Baseten](https://app.baseten.co/models/).

### 3. Invoke the Model

After deployment, test your model:

```bash
truss predict -d '"Truss is awesome!"'
```

**Response**

```json
[
  {
    "label": "POSITIVE",
    "score": 0.999873161315918
  }
]
```

## Truss Contributors

Truss is supported by Baseten and benefits from the contributions of ML engineers worldwide. Special thanks to [Stephan Auerhahn](https://github.com/palp) @ [stability.ai](https://stability.ai/) and [Daniel Sarfati](https://github.com/dsarfati) @ [Salad Technologies](https://salad.com/) for their contributions.

We welcome contributions! See our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).