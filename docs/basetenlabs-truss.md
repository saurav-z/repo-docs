# Truss: The Simple Way to Deploy and Serve Your AI/ML Models

**[Get started with Truss on GitHub](https://github.com/basetenlabs/truss)** and effortlessly deploy your machine learning models to production with ease.

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features of Truss

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Rapid Development:** Experience a fast developer loop with a live reload server, eliminating the need for Docker and Kubernetes configuration.
*   **Framework Agnostic:** Seamlessly supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models with a single command using Baseten (with AWS SageMaker coming soon).
*   **Easy Model Invocation:** Quickly test and interact with your deployed models through a straightforward command-line interface.

## Ready-to-Use Examples

Explore pre-built Trusses for popular models:

*   🦙 [Llama 2 (7B, 13B, 70B)](https://github.com/basetenlabs/truss-examples/tree/main/llama)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

Discover [dozens more examples](https://github.com/basetenlabs/truss-examples/) to jumpstart your model deployment.

## Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

## Quickstart Guide

Follow these steps to package and deploy a text classification pipeline using `transformers`:

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Name your Truss, for instance, `Text classification`.
2.  **Navigate to the Truss directory:**

    ```bash
    cd text-classification
    ```
3.  **Implement the Model:**
    *   Edit `model/model.py` to define your model's `load()` and `predict()` methods. Here's the text classification example:

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
    *   Modify `config.yaml` to include your model's dependencies.  For the text classification example, add `transformers` and `torch`:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment to Baseten

1.  **Get an API Key:**
    *   Obtain a [Baseten API key](https://app.baseten.co/settings/account/api_keys).  [Sign up for a Baseten account](https://app.baseten.co/signup/) if you don't have one.
2.  **Deploy with `truss push`:**

    ```bash
    truss push
    ```

    Monitor your deployment through the [Baseten model dashboard](https://app.baseten.co/models/).
3.  **Invoke the Model:**
    *   After deployment, use the following command to test your model:

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

## Contributing

Truss is a community project backed by Baseten, and welcomes contributions!  Refer to the [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).