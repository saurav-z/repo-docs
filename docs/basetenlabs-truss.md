# Truss: The Simplest Way to Deploy AI/ML Models in Production

**Quickly and easily deploy your machine learning models with Truss, a powerful open-source framework.** ([View on GitHub](https://github.com/basetenlabs/truss))

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Fast Developer Loop:** Leverage a live reload server for rapid iteration, eliminating the need for complex Docker or Kubernetes configurations.
*   **Framework Agnostic:** Supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Easily deploy your models to Baseten (with more platforms coming soon, like AWS SageMaker).
*   **Built-in Configuration:**  Manage your model's environment with a simple configuration file (`config.yaml`).

## Getting Started

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification Example

Follow these steps to deploy a text classification model using the `transformers` library.

**1. Create a Truss:**

```bash
truss init text-classification
```

Give your Truss a name when prompted.

**2. Navigate to the new directory:**

```bash
cd text-classification
```

**3. Implement the Model (`model/model.py`):**

Create a `Model` class to interface with the model server. Implement the `load()` and `predict()` methods:

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

**4. Add Model Dependencies (`config.yaml`):**

Specify dependencies in `config.yaml`:

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

## Deployment with Baseten

Truss seamlessly integrates with Baseten for production deployment.

**1. Get a Baseten API Key:**

Sign up for a Baseten account ([https://app.baseten.co/signup/](https://app.baseten.co/signup/)) and obtain your API key ([https://app.baseten.co/settings/account/api_keys](https://app.baseten.co/settings/account/api_keys)).

**2. Push Your Model:**

```bash
truss push
```

Follow the prompts to deploy your model to Baseten. Monitor the deployment from your Baseten model dashboard ([https://app.baseten.co/models/](https://app.baseten.co/models/)).

**3. Invoke the Model:**

Once deployed, test your model from the terminal:

```bash
truss predict -d '"Truss is awesome!"'
```

**Example Response:**

```json
[
  {
    "label": "POSITIVE",
    "score": 0.999873161315918
  }
]
```

## Examples

Truss supports a wide range of models.  Explore these examples:

*   🦙 [Llama 2 (7B, 13B, 70B)](https://github.com/basetenlabs/truss-examples/tree/main/llama)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

... and [dozens more](https://github.com/basetenlabs/truss-examples/)!

## Contributing

Truss is backed by Baseten and built in collaboration with ML engineers worldwide.  We welcome contributions!  See the [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).