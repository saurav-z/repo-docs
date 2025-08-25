# Truss: The Fastest Way to Serve Your AI/ML Models

**Effortlessly deploy and manage your machine learning models in production with Truss.**

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![CI Status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a standardized format that runs consistently across development and production environments.
*   **Rapid Development Loop:** Utilize a live reload server for immediate feedback and streamline your workflow, eliminating the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Truss supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Easily deploy your models to Baseten (with more remotes coming soon) with a single command.

## Popular Model Examples

Get started quickly with pre-configured Trusses for popular models:

*   🦙 [Llama 2 (7B, 13B, 70B)](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

Explore [dozens more examples](https://github.com/basetenlabs/truss-examples/) to find the perfect starting point for your project.

## Getting Started

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification

Follow these steps to package and deploy a text classification model using Hugging Face's Transformers library:

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Enter a name for your Truss (e.g., "Text classification").
2.  **Navigate to the New Directory:**

    ```bash
    cd text-classification
    ```
3.  **Implement the Model (`model/model.py`):**

    Create a `Model` class with `load()` (for model initialization) and `predict()` (for inference) methods:

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

    Specify your model's dependencies in the `config.yaml` file under the `requirements` section.  For the text classification example, update the `requirements` section to:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```
5.  **Deployment**

    Truss is currently integrated with Baseten, which provides infrastructure for running ML models in production.

    *   **Get a Baseten API Key:** Sign up for an account at [https://app.baseten.co/signup/](https://app.baseten.co/signup/) and obtain your API key from [https://app.baseten.co/settings/account/api_keys](https://app.baseten.co/settings/account/api_keys).
    *   **Deploy with `truss push`:**

        ```bash
        truss push
        ```

        Paste your Baseten API key when prompted.  Monitor your model deployment at [your model dashboard on Baseten](https://app.baseten.co/models/).
    *   **Invoke the Model:**
        Once the model is deployed, you can test it from the command line:

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

We welcome contributions! Please refer to our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

## Resources

*   **[Truss GitHub Repository](https://github.com/basetenlabs/truss)**
*   **[Baseten](https://baseten.co)**