# Truss: The Simplest Way to Deploy & Serve Your AI/ML Models in Production

**Quickly and easily deploy your machine learning models with Truss, a powerful tool for packaging, serving, and managing AI/ML models in production.**

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![CI Status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features of Truss

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a consistent environment that behaves the same in development and production.
*   **Accelerated Development Loop:** Utilize a live reload server for rapid feedback and skip complex Docker and Kubernetes configurations, streamlining your development workflow.
*   **Framework Agnostic:** Truss seamlessly supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Easily deploy your models to production environments, starting with Baseten and AWS SageMaker support coming soon.
*   **Easy Model Invocation:** Simple CLI commands for testing and interacting with your deployed models.

## Explore Truss Examples

Get started quickly with pre-built Trusses for popular models:

*   🦙 [Llama 2](https://github.com/basetenlabs/truss-examples/tree/main/llama) (7B, 13B, and 70B variations)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

Find even more examples in the [Truss Examples repository](https://github.com/basetenlabs/truss-examples/).

## Getting Started

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Deploying a Text Classification Model

This quickstart example demonstrates how to deploy a text classification pipeline using the `transformers` library.

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Give your Truss a name, such as "Text classification."

2.  **Navigate to the Directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement the Model:**
    Edit the `model/model.py` file to define your model's `load()` and `predict()` functions:

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

4.  **Add Dependencies:**
    Modify the `config.yaml` file to include the necessary dependencies:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment with Baseten

1.  **Get a Baseten API Key:**
    If you don't have one, [sign up for a Baseten account](https://app.baseten.co/signup/) and obtain your API key from your [Baseten account settings](https://app.baseten.co/settings/account/api_keys).

2.  **Deploy Your Model:**
    Run the following command in your terminal to deploy your model:

    ```bash
    truss push
    ```

    Monitor your model's deployment progress via your [Baseten model dashboard](https://app.baseten.co/models/).

3.  **Invoke the Model:**
    Once deployed, test your model using the `truss predict` command:

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```

    You should receive a JSON response similar to:

    ```json
    [
      {
        "label": "POSITIVE",
        "score": 0.999873161315918
      }
    ]
    ```

## Contributing

Truss is an open-source project backed by Baseten, built with the help of ML engineers worldwide. We welcome contributions! Please see our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

**Original repository:** [https://github.com/basetenlabs/truss](https://github.com/basetenlabs/truss)