# Truss: The Easiest Way to Deploy and Serve Your Machine Learning Models

**[Original Repository](https://github.com/basetenlabs/truss)**

Truss simplifies the process of taking your AI/ML models from development to production with ease.

## Key Features

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies seamlessly for consistent behavior across development and production environments.
*   **Accelerated Development:** Benefit from a fast developer loop with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models with a single command using Baseten, with support for AWS SageMaker coming soon.
*   **Example Models:** Ready-to-use examples for popular models like Llama 2, Stable Diffusion XL, and Whisper, and many more.

## Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

## Quickstart: Deploying a Text Classification Model

Follow these steps to get started with a text classification model using the `transformers` library:

### 1. Create a Truss

Use the following command in your terminal to initialize a new Truss:

```bash
truss init text-classification
```

Provide a name for your Truss when prompted, such as `Text classification`.

Navigate into the newly created directory:

```bash
cd text-classification
```

### 2. Implement the Model (`model/model.py`)

Create the `model/model.py` file and define a `Model` class with `load()` and `predict()` methods:

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

Configure dependencies in `config.yaml`. Replace the `requirements` section with:

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

## Deployment

Truss is maintained by [Baseten](https://baseten.co), providing the infrastructure for running ML models in production.

### Get an API Key

1.  Get a [Baseten API key](https://app.baseten.co/settings/account/api_keys).  If you don't have an account, [sign up](https://app.baseten.co/signup/) for free credits.

### Deploy with `truss push`

Deploy your model using the following command, entering your Baseten API key when prompted:

```bash
truss push
```

Monitor your model deployment from [your model dashboard on Baseten](https://app.baseten.co/models/).

### Invoke the Model

Once deployed, test your model from the terminal.

**Invocation**

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

## Contributions

Truss is backed by Baseten and welcomes contributions.  See our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md) for more information.