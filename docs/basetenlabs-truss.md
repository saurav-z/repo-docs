# Truss: Deploy AI/ML Models in Production with Ease

**Truss simplifies AI/ML model deployment, allowing you to effortlessly package, serve, and scale your models in production.**  [Explore the original repository](https://github.com/basetenlabs/truss).

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features of Truss:

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Rapid Development Loop:** Benefit from a live reload server for fast feedback and streamline your workflow, eliminating the need for complex Docker and Kubernetes configurations.
*   **Broad Framework Support:** Truss seamlessly integrates with any Python-based ML framework, including Transformers, TensorFlow, PyTorch, TensorRT, and Triton.
*   **Easy Deployment:** Deploy your models with `truss push` and monitor them with the Baseten platform.
*   **Production-Ready:** Features like model versioning, monitoring, and auto-scaling come out of the box.

## Getting Started: Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

## Quickstart: Deploying a Text Classification Model

This quickstart demonstrates how to package and deploy a text classification pipeline using Truss and the Hugging Face `transformers` library.

### 1. Create a Truss

Generate a new Truss project:

```bash
truss init text-classification
```

Follow the prompts to name your Truss (e.g., "Text classification").

```bash
cd text-classification
```

### 2. Implement the Model

Create the core model logic in `model/model.py`:

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

### 3. Define Dependencies

Specify project dependencies (Transformers and PyTorch) in `config.yaml`:

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

## Deployment: Serving Your Model

Truss integrates with Baseten for model deployment.

### 1. Obtain a Baseten API Key

Sign up for a free account and obtain an API key from [Baseten](https://app.baseten.co/signup/).

### 2. Deploy Your Model

Push your model to Baseten:

```bash
truss push
```

Monitor the deployment through your [Baseten model dashboard](https://app.baseten.co/models/).

### 3. Invoke the Model

After deployment, test the model:

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

## Truss Examples

*   🦙 [Llama 2 7B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) ([13B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-13b-chat)) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-70b-chat))
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

and [dozens more examples](https://github.com/basetenlabs/truss-examples/).

## Contributing

We welcome contributions! Please see the [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

## Acknowledgements

Special thanks to [Stephan Auerhahn](https://github.com/palp) @ [stability.ai](https://stability.ai/) and [Daniel Sarfati](https://github.com/dsarfati) @ [Salad Technologies](https://salad.com/) for their contributions.