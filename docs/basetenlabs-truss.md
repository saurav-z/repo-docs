# Truss: The Easiest Way to Deploy & Serve Your Machine Learning Models

**Truss** simplifies the process of deploying and serving your machine learning models, offering a streamlined experience for developers. Check out the original repo on GitHub: [basetenlabs/truss](https://github.com/basetenlabs/truss).

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features

*   **Write Once, Run Anywhere:** Package and test your model code, weights, and dependencies in a consistent environment for development and production.
*   **Rapid Development Loop:** Benefit from a live reload server for quick feedback and bypass complex Docker/Kubernetes configurations.
*   **Framework Agnostic:** Truss supports models built with any Python framework, including `transformers`, `diffusers`, PyTorch, TensorFlow, TensorRT, and Triton.
*   **Simplified Deployment:** Easily deploy your models to Baseten (with more remotes coming soon) with a single command.
*   **Pre-built Examples:** Get started quickly with Truss examples for popular models such as Llama 2, Stable Diffusion XL, and Whisper.

## Get Started with Truss

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification Example

This quickstart demonstrates how to package and deploy a text classification pipeline using the `transformers` library.

1.  **Create a Truss:**
    ```bash
    truss init text-classification
    ```
    Give your Truss a name, like "Text classification."

2.  **Navigate to the directory:**
    ```bash
    cd text-classification
    ```

3.  **Implement the Model (`model/model.py`):**
    Create a `Model` class with `load()` and `predict()` functions:

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

4.  **Add Model Dependencies (`config.yaml`):**
    Specify dependencies in the `config.yaml` file:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment

Truss is designed to be deployed on the Baseten platform.

### Deploying to Baseten

1.  **Get a Baseten API Key:**  Sign up for a free Baseten account and obtain your API key.
    ( [https://app.baseten.co/signup/](https://app.baseten.co/signup/) )
2.  **Push Your Model:**
    ```bash
    truss push
    ```
    Follow the prompts, providing your Baseten API key.  Monitor your deployment from your [Baseten model dashboard](https://app.baseten.co/models/).

### Invoking Your Deployed Model

Once deployed, test your model:

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

## Truss Contributors

Truss is backed by Baseten and developed in collaboration with the ML community. Contributions are welcome; please review our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).