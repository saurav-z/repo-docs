# Truss: The Easiest Way to Deploy and Serve Your AI/ML Models

**Truss** simplifies the process of deploying and managing your machine learning models, enabling you to focus on building and innovating. Find the original repo [here](https://github.com/basetenlabs/truss).

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Fast Developer Loop:** Experience rapid iteration with a live reload server, eliminating the need for complex Docker or Kubernetes configurations.
*   **Framework Agnostic:** Truss supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Streamlined Deployment:** Deploy your models effortlessly to Baseten and, soon, AWS SageMaker, with more remote options coming soon.
*   **Quickstart Example:** Package and serve a text classification pipeline from the open-source `transformers` package.

## Supported Models & Examples

Truss provides pre-built examples for popular models, including:

*   🦙 [Llama 2 7B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) ([13B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-13b-chat)) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-70b-chat))
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

Explore [dozens more examples](https://github.com/basetenlabs/truss-examples/) to get started quickly.

## Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

## Quickstart Guide: Deploying a Text Classification Model

This quickstart demonstrates how to package and deploy a text classification pipeline using `transformers`.

### 1. Create a Truss

Initiate a new Truss project with the following command:

```bash
truss init text-classification
```

Name your Truss project (e.g., "Text classification") when prompted.

### 2. Navigate into the Project Directory

Change your current directory into the newly created project folder.

```bash
cd text-classification
```

### 3. Implement Your Model

Edit `model/model.py` to define a `Model` class with `load()` and `predict()` methods:

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

### 4. Add Model Dependencies

Configure your model's dependencies in `config.yaml`.  Specify the required packages:

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

## Deployment

Truss models are currently deployed via [Baseten](https://baseten.co). AWS SageMaker deployment is coming soon.

### 1. Get a Baseten API Key

Create or retrieve your Baseten API key from your [Baseten account settings](https://app.baseten.co/settings/account/api_keys).  If you need an account, [sign up](https://app.baseten.co/signup/) to get started.

### 2. Deploy with `truss push`

Deploy your model using the command:

```bash
truss push
```

Monitor the deployment progress from [your model dashboard on Baseten](https://app.baseten.co/models/).

### 3. Invoke Your Deployed Model

Test your model by sending an input:

```bash
truss predict -d '"Truss is awesome!"'
```

**Expected Response:**

```json
[
  {
    "label": "POSITIVE",
    "score": 0.999873161315918
  }
]
```

## Contributing

Truss is a community-driven project backed by Baseten. We welcome contributions! Please consult our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).