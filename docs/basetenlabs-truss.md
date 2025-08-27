# Truss: The Simple Way to Serve Your AI/ML Models

**Easily deploy and manage your machine learning models in production with Truss**, a powerful framework designed to simplify the model serving process.

[View the original Truss repository on GitHub](https://github.com/basetenlabs/truss)

## Key Features of Truss

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Rapid Development:** Iterate quickly with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Truss seamlessly supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models to Baseten with a simple `truss push` command, and other remotes are coming soon.
*   **Built-in Examples:** Get started quickly with pre-built Truss examples for popular models like Llama 2, Stable Diffusion XL, and Whisper.

## Getting Started

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification Example

1.  **Initialize a Truss:**

    ```bash
    truss init text-classification
    ```
    Give your Truss a name, such as "Text classification".

2.  **Navigate to the directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement the Model (`model/model.py`):**
    Create a `Model` class with `load()` and `predict()` methods.

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

4.  **Define Dependencies (`config.yaml`):**
    Specify your model's dependencies in `config.yaml`. For the text classification example:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment

Truss is currently maintained by [Baseten](https://baseten.co).

### 1. Get a Baseten API Key

Sign up for a [Baseten account](https://app.baseten.co/signup/) and obtain an API key from your account settings.

### 2. Deploy with `truss push`

```bash
truss push
```

Monitor the deployment progress on your [Baseten model dashboard](https://app.baseten.co/models/).

### 3. Invoke Your Model

After deployment, test your model:

**Invocation:**

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

## Truss Contributors

Truss is actively developed by Baseten in collaboration with ML engineers. Special thanks to Stephan Auerhahn and Daniel Sarfati for their contributions. Contributions are welcome, please refer to the [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).