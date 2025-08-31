# Truss: The Simplest Way to Deploy and Serve Your AI/ML Models

**Effortlessly deploy and manage your machine learning models in production with Truss, the open-source framework that simplifies model serving.** ([Original Repo](https://github.com/basetenlabs/truss))

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features of Truss

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a portable Truss that behaves consistently across development and production environments.
*   **Rapid Development Cycle:** Benefit from a fast developer loop with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   **Broad Framework Support:** Truss seamlessly integrates with all major Python ML frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models to various platforms with ease, starting with Baseten, and soon expanding to include AWS SageMaker.
*   **Pre-Built Examples:** Get started quickly with ready-to-use Truss examples for popular models like Llama 2, Stable Diffusion XL, and Whisper.

## Getting Started with Truss

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Deploy a Text Classification Model

This quickstart demonstrates how to package and deploy a text classification pipeline using the open-source `transformers` library.

1.  **Create a Truss:**
    ```bash
    truss init text-classification
    ```
    Name your Truss, e.g., "Text classification".

2.  **Navigate to the Truss directory:**
    ```bash
    cd text-classification
    ```

3.  **Implement the Model:**
    In `model/model.py`, define a `Model` class with `load()` (model loading) and `predict()` (inference) methods:

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

4.  **Add Model Dependencies:**
    Specify dependencies (e.g., Transformers and PyTorch) in `config.yaml`:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment

Truss simplifies model deployment through Baseten.

1.  **Get an API Key:**
    Obtain a [Baseten API key](https://app.baseten.co/settings/account/api_keys) (sign up [here](https://app.baseten.co/signup/) if you don't have an account).

2.  **Deploy Your Model:**
    ```bash
    truss push
    ```
    Follow the prompts, providing your Baseten API key. Monitor deployment from [your Baseten model dashboard](https://app.baseten.co/models/).

3.  **Invoke the Model:**
    After deployment, test your model:

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```
    You should receive a JSON response indicating a positive sentiment.

## Examples

Explore pre-built Trusses for popular models:

*   🦙 [Llama 2 7B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) ([13B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-13b-chat)) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-70b-chat))
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

And [dozens more examples](https://github.com/basetenlabs/truss-examples/).

## Contribute

Truss welcomes contributions! Review the [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

**Thanks to our contributors:** [Stephan Auerhahn](https://github.com/palp) @ [stability.ai](https://stability.ai/) and [Daniel Sarfati](https://github.com/dsarfati) @ [Salad Technologies](https://salad.com/)