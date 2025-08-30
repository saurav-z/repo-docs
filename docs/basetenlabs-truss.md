# Truss: The Easiest Way to Deploy and Serve Your Machine Learning Models

**Truss** ([See original repo](https://github.com/basetenlabs/truss)) is a powerful and user-friendly framework designed to streamline the deployment and serving of your AI/ML models in production. 

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features of Truss:

*   **Simplified Deployment:** Package your model code, dependencies, and weights into a standardized format for easy deployment.
*   **Write Once, Run Anywhere:** Develop and test your model locally with the same environment it will use in production.
*   **Fast Iteration:** Benefit from a live reload server for rapid feedback during development, eliminating the need for complex Docker or Kubernetes configurations.
*   **Framework Agnostic:** Seamlessly supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Easy Integration:** Quickly deploy and manage models on various platforms, including Baseten (with AWS SageMaker support coming soon).

## Get Started with Truss:

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart Example: Text Classification

This quickstart demonstrates how to package a text classification pipeline from the Hugging Face `transformers` library.

1.  **Create a Truss:**
    ```bash
    truss init text-classification
    ```

2.  **Navigate to the New Directory:**
    ```bash
    cd text-classification
    ```

3.  **Implement the Model (`model/model.py`):**
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
    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

### Deployment using Baseten:

1.  **Get a Baseten API Key:**  Sign up for a free account at [Baseten](https://app.baseten.co/signup/) and obtain your API key from [your account settings](https://app.baseten.co/settings/account/api_keys).
2.  **Deploy Your Model:**
    ```bash
    truss push
    ```
3.  **Invoke the Model:**
    ```bash
    truss predict -d '"Truss is awesome!"'
    ```

## Example Models

Truss supports various popular models. Here are a few examples:

*   🦙 [Llama 2 7B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) ([13B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-13b-chat)) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-70b-chat))
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

And [dozens more examples](https://github.com/basetenlabs/truss-examples/).

## Contribute

Truss is an open-source project backed by Baseten and built in collaboration with the ML community.  We welcome contributions! Please refer to our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).