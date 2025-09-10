# Truss: The Easiest Way to Serve Your AI/ML Models

**[View the original repository on GitHub](https://github.com/basetenlabs/truss)**

Truss simplifies the deployment and management of your AI/ML models, allowing you to focus on building and innovating. Package your models with ease and deploy them seamlessly, no matter the framework.

## Key Features

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies once, and run it consistently across development and production environments.
*   **Rapid Development:** Utilize a live reload server for fast feedback, streamlining your development workflow, and reducing the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Supports all major Python frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Pre-built Examples:** Quickly deploy popular models like Llama 2, Stable Diffusion XL, and Whisper with available examples and numerous other pre-built examples.

## Getting Started

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Deploy a Text Classification Model

Follow these steps to quickly deploy a text classification pipeline using the open-source `transformers` package.

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Provide a name for your Truss.

2.  **Navigate to the Directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement Your Model (`model/model.py`):**

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

4.  **Add Dependencies (`config.yaml`):**

    Specify dependencies in the `config.yaml` file. Replace the `requirements` section with:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment

Truss is maintained by [Baseten](https://baseten.co), offering infrastructure for running ML models in production.

1.  **Get a Baseten API Key:**

    Sign up for a [Baseten account](https://app.baseten.co/signup/) and obtain your API key from [Baseten API keys settings](https://app.baseten.co/settings/account/api_keys).

2.  **Run `truss push`:**

    ```bash
    truss push
    ```

    Monitor your model deployment from [your model dashboard on Baseten](https://app.baseten.co/models/).

### Invoke the Model

After deployment, invoke your model from the terminal:

**Invocation**

```bash
truss predict -d '"Truss is awesome!"'
```

**Response**

```json
[
  {
    "label": "POSITIVE",
    "score": 0.999873161315918
  }
]
```

## Contributing

We welcome contributions! Please see our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).