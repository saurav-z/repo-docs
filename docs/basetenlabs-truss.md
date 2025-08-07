# Truss: Deploy AI/ML Models Simply & Efficiently

**Truss is a streamlined framework for effortlessly packaging, serving, and deploying your machine learning models in production.**

[View the original repository on GitHub](https://github.com/basetenlabs/truss)

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies with a consistent model server environment for both development and production.
*   **Rapid Development Cycle:** Leverage a live reload server for quick feedback and skip complex Docker and Kubernetes configurations.
*   **Comprehensive Framework Support:** Truss seamlessly supports models from all major Python ML frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Easily deploy your models using Baseten's infrastructure.
*   **Pre-Built Examples:** Start quickly with example Trusses for popular models like Llama 2, Stable Diffusion XL, and Whisper.

## Getting Started

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Deploy a Text Classification Model

This quickstart demonstrates how to package and deploy a text classification model using the Hugging Face `transformers` library.

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Provide a name for your Truss (e.g., "Text classification").

2.  **Navigate to the Directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement the Model (`model/model.py`):**

    Create a `Model` class with `load()` and `predict()` methods:

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

    Specify required packages in `config.yaml`:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment (Using Baseten)

1.  **Get a Baseten API Key:** Obtain an API key from your [Baseten account](https://app.baseten.co/settings/account/api_keys). If you don't have an account, [sign up](https://app.baseten.co/signup/).

2.  **Deploy Your Model:**

    ```bash
    truss push
    ```

    Follow the prompts to configure your deployment, providing your Baseten API key.

3.  **Invoke the Model:** After the deployment is complete, invoke your model.

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

## Contributing

Truss is a collaborative project backed by Baseten and contributions from the global ML engineering community. Contributions are welcome in accordance with our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).