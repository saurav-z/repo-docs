# Truss: The Easiest Way to Deploy and Serve Your Machine Learning Models

**Truss** is a powerful open-source framework that simplifies the process of packaging, deploying, and serving your machine learning models in production.  [Learn more on GitHub](https://github.com/basetenlabs/truss).

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features & Benefits

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a consistent environment that works the same in development and production.
*   **Fast Development Loop:**  Utilize a live reload server for rapid iteration and skip complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Easily deploy your models to various platforms, starting with Baseten, and future support for AWS SageMaker.
*   **Pre-built Examples:** Get started quickly with pre-configured Trusses for popular models like Llama 2, Stable Diffusion XL, and Whisper, plus dozens more.

## Get Started Quickly

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Deploy a Text Classification Model

Follow these steps to create and deploy a simple text classification model using the `transformers` library.

1.  **Initialize a Truss:**
    ```bash
    truss init text-classification
    ```
    Give your Truss a name, such as `Text classification`.
2.  **Navigate to the new directory:**
    ```bash
    cd text-classification
    ```
3.  **Implement your Model (`model/model.py`):** Create a `Model` class with `load()` and `predict()` functions.  Here's the example for the text classification model:

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
4.  **Configure Dependencies (`config.yaml`):** Specify your model's dependencies in `config.yaml`:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```
5.  **Deploy Your Model:**

    *   Get a [Baseten API key](https://app.baseten.co/settings/account/api_keys) (or [sign up](https://app.baseten.co/signup/)).
    *   Run `truss push` and provide your API key when prompted.

    ```bash
    truss push
    ```
    Monitor the deployment in your [Baseten model dashboard](https://app.baseten.co/models/).
6.  **Invoke Your Model:** Once deployed, test your model from the terminal:
    ```bash
    truss predict -d '"Truss is awesome!"'
    ```
    You'll receive a JSON response like this:
    ```json
    [
      {
        "label": "POSITIVE",
        "score": 0.999873161315918
      }
    ]
    ```

## Community and Contributions

Truss is backed by Baseten and built with the help of ML engineers. We welcome contributions! See our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).