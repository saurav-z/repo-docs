# Truss: Effortless Model Serving for AI/ML in Production

**Truss simplifies the deployment of your machine learning models, enabling you to focus on innovation, not infrastructure.**

[View the original repository](https://github.com/basetenlabs/truss)

## Key Features

*   **"Write Once, Run Anywhere"**: Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Fast Development Cycle**: Benefit from a live reload server for rapid feedback and eliminate complex Docker and Kubernetes configurations.
*   **Comprehensive Framework Support**: Truss seamlessly integrates with all major Python ML frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and Triton.
*   **Simplified Deployment**: Easily deploy your models to Baseten (with AWS SageMaker support coming soon) with a single command.
*   **Scalable & Production-Ready**: Designed for production environments, allowing you to scale your models efficiently.

## Examples & Use Cases

Truss supports a wide range of popular models and use cases. Explore these examples to get started:

*   **Large Language Models (LLMs)**:
    *   Llama 2 (7B, 13B, 70B)
*   **Image Generation**:
    *   Stable Diffusion XL
*   **Speech Recognition**:
    *   Whisper
*   And [dozens more examples](https://github.com/basetenlabs/truss-examples/).

## Installation

Get started with Truss by installing it via pip:

```bash
pip install --upgrade truss
```

## Quickstart: Text Classification Model

Follow these simple steps to deploy a text classification model using Truss:

### 1. Create a Truss

```bash
truss init text-classification
```

Give your Truss a name, like "Text classification".

```bash
cd text-classification
```

### 2. Implement Your Model in `model/model.py`

Define a `Model` class with `load()` (loads the model) and `predict()` (handles inference) functions:

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

### 3. Add Dependencies in `config.yaml`

Specify your model's dependencies in `config.yaml`. For example:

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

## Deployment to Baseten

1.  **Get a Baseten API Key**:  Sign up for a free account on [Baseten](https://app.baseten.co/signup/) and obtain your API key from your account settings ([https://app.baseten.co/settings/account/api_keys](https://app.baseten.co/settings/account/api_keys)).
2.  **Deploy Your Model**:

    ```bash
    truss push
    ```

    Monitor your model deployment on your [Baseten model dashboard](https://app.baseten.co/models/).

3.  **Invoke Your Model**:

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

## Contributing

Truss is an open-source project backed by Baseten, with contributions from ML engineers worldwide. We welcome contributions following our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).