# Truss: The Easiest Way to Deploy Your Machine Learning Models

**Truss** simplifies the process of deploying and serving your AI/ML models in production, offering a streamlined experience for developers. ([Original Repository](https://github.com/basetenlabs/truss))

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![CI Status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features of Truss

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies with a consistent model server environment that behaves the same in development and production.
*   **Rapid Development Cycle:** Benefit from a fast developer loop with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   **Comprehensive Framework Support:** Truss seamlessly supports models built using any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models with ease using Baseten, with support for AWS SageMaker and more remotes coming soon.

## Examples of Models Deployed with Truss

*   🦙 [Llama 2 7B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) ([13B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-13b-chat)) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-70b-chat))
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

And explore [dozens more examples](https://github.com/basetenlabs/truss-examples/) to get started quickly.

## Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

## Quickstart: Text Classification Example

This quickstart demonstrates how to package a text classification pipeline using the `transformers` library.

### Create a Truss

Initialize a Truss project:

```bash
truss init text-classification
```

### Implement the Model

Edit `model/model.py` to define your model class, including `load()` and `predict()` methods.

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

### Add Model Dependencies

Specify dependencies in `config.yaml`:

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

## Deployment

Deploy your model to Baseten.

### Get a Baseten API Key

Sign up for a Baseten account and obtain your API key ([https://app.baseten.co/settings/account/api_keys](https://app.baseten.co/settings/account/api_keys)).

### Push Your Model

Deploy your model using:

```bash
truss push
```

Monitor deployment on your Baseten dashboard ([https://app.baseten.co/models/](https://app.baseten.co/models/)).

### Invoke the Model

Test your deployed model:

**Invocation:**

```bash
truss predict -d '"Truss is awesome!"'
```

**Response:**

```json
[
  {
    "label": "POSITIVE",
    "score": 0.999873161315918
  }
]
```

## Contributing

Truss welcomes contributions!  See our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).