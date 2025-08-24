# Truss: The Easiest Way to Deploy Your AI/ML Models

**Truss** simplifies the process of deploying machine learning models, enabling you to get your models into production quickly and efficiently. Learn more at the [original repo](https://github.com/basetenlabs/truss).

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features of Truss:

*   **Write Once, Run Anywhere:** Package and test your model code, weights, and dependencies, ensuring consistent behavior across development and production environments.
*   **Fast Developer Loop:** Benefit from a live reload server for rapid iteration and feedback, eliminating the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Truss supports models built with any Python framework, including Transformers, Diffusers, PyTorch, TensorFlow, TensorRT, and Triton.
*   **Simplified Deployment:** Deploy models to platforms like Baseten (with AWS SageMaker support coming soon) with a single command.
*   **Ready-to-Use Examples:** Get started quickly with pre-built Truss examples for popular models like Llama 2, Stable Diffusion XL, and Whisper.

## Get Started with Truss

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Deploy a Text Classification Model

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

2.  **Navigate to the Directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement Your Model (model/model.py):**

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

4.  **Add Dependencies (config.yaml):**

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

### Deployment

1.  **Get a Baseten API Key:**
    Sign up for a [Baseten account](https://app.baseten.co/signup/) to obtain an API key.
2.  **Deploy with `truss push`:**

    ```bash
    truss push
    ```

3.  **Invoke the Model:**

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

## Contributions

Truss is an open-source project backed by Baseten, built with contributions from the ML community. Contributions are welcome; please refer to our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).