# Truss: Deploy AI/ML Models with Ease

**Truss simplifies the process of deploying and serving your machine learning models in production, offering a streamlined experience for developers.** ([Original Repo](https://github.com/basetenlabs/truss))

## Key Features

*   **Write Once, Run Anywhere:** Package your model code, dependencies, and weights in a consistent environment for seamless deployment across different platforms.
*   **Fast Developer Loop:** Iterate quickly with a live reload server, enabling rapid testing and feedback without complex Docker or Kubernetes configurations.
*   **Framework Agnostic:** Truss supports a wide range of Python ML frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Pre-built Examples:** Get started quickly with examples for popular models like Llama 2, Stable Diffusion XL, and Whisper.

## Getting Started

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification

This example demonstrates how to deploy a text classification pipeline using the `transformers` library.

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Name your Truss (e.g., "Text classification").

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

    In `config.yaml`, add the following under the `requirements` section:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment

Truss integrates with Baseten for production model serving. Other remotes, like AWS SageMaker, are planned.

1.  **Get a Baseten API Key:**  Sign up for a free account at [Baseten](https://app.baseten.co/signup/) and obtain your API key from [your account settings](https://app.baseten.co/settings/account/api_keys).

2.  **Push Your Model:**

    ```bash
    truss push
    ```

    Monitor your model deployment on the [Baseten model dashboard](https://app.baseten.co/models/).

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

## Contributing

We welcome contributions!  See our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).