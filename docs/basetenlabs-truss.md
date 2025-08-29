# Truss: The Simplest Way to Deploy and Serve Your AI/ML Models in Production

**[View the original repository on GitHub](https://github.com/basetenlabs/truss)**

Truss simplifies the process of deploying and serving your machine learning models, enabling you to move from development to production quickly and efficiently.

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features of Truss

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a consistent environment that behaves the same in development and production.
*   **Rapid Development Cycle:** Benefit from a fast feedback loop with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Truss supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Easy Deployment:** Deploy your models to Baseten and other platforms (like AWS SageMaker in the future) with a simple command.
*   **Simplified Model Serving:** Focus on your model logic, not infrastructure, with a streamlined model serving environment.

## Popular Model Examples with Truss

Get started quickly with pre-configured Truss examples for popular models:

*   🦙 [Llama 2](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) (7B, 13B, and 70B)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

And explore [dozens more examples](https://github.com/basetenlabs/truss-examples/) to fit a variety of use cases.

## Getting Started with Truss

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification Example

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Give your Truss a name, such as "Text classification".
2.  **Navigate to the New Directory:**

    ```bash
    cd text-classification
    ```
3.  **Implement Your Model (`model/model.py`):**  Define a `Model` class with `load()` (loads the model) and `predict()` (handles inference) methods.  For example:

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

4.  **Add Dependencies (`config.yaml`):**  Specify your model's dependencies.  For the text classification example, modify `config.yaml`:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment

Truss is backed by [Baseten](https://baseten.co), which provides the infrastructure for running ML models in production.

### Get Your Baseten API Key

Sign up for a [Baseten account](https://app.baseten.co/signup/) and obtain your API key from [your account settings](https://app.baseten.co/settings/account/api_keys).

### Deploy Your Model

Deploy your model to Baseten with:

```bash
truss push
```

Monitor your deployment in your [Baseten model dashboard](https://app.baseten.co/models/).

### Invoke the Model

Once deployed, test your model from the terminal:

```bash
truss predict -d '"Truss is awesome!"'
```

## Truss Contributors

Truss is built in collaboration with ML engineers worldwide and backed by Baseten.

## Contribute

We welcome contributions!  Please review our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).