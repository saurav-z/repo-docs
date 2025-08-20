# Truss: The Simplest Way to Deploy Your AI/ML Models

**Truss** is an open-source framework that simplifies deploying and serving your AI/ML models in production, enabling you to focus on building and iterating. Learn more on [GitHub](https://github.com/basetenlabs/truss).

## Key Features

*   🚀 **Write Once, Run Anywhere:** Package your model code, weights, and dependencies with a consistent model server environment, simplifying deployment across development and production.
*   ⚡️ **Fast Development Loop:** Enjoy rapid feedback with a live reload server, streamlining your model development process and eliminating the need for complex Docker/Kubernetes configurations.
*   🛠️ **Framework Agnostic:** Truss supports a wide range of Python frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`, accommodating diverse model implementations.
*   ✅ **Pre-built Examples**: Get started quickly with pre-built "Trusses" for popular models like Llama 2, Stable Diffusion XL, and Whisper.

## Get Started with Truss

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Deploy a Text Classification Model

Follow these steps to quickly deploy a text classification model using the `transformers` library.

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

2.  **Navigate to the Directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement Your Model (`model/model.py`):**

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

4.  **Configure Dependencies (`config.yaml`):**
    Update the `requirements` section in `config.yaml` to include:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

### Deployment

Truss integrates with [Baseten](https://baseten.co) for production model serving, with more deployment options coming soon.

1.  **Get a Baseten API Key:** Sign up for a free account at [Baseten](https://app.baseten.co/signup/) and obtain your API key from your account settings.
2.  **Deploy Your Model:**

    ```bash
    truss push
    ```
    Monitor your deployment on your [Baseten model dashboard](https://app.baseten.co/models/).

3.  **Invoke the Model:**

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```
    Example response:

    ```json
    [
      {
        "label": "POSITIVE",
        "score": 0.999873161315918
      }
    ]
    ```

## Contribute

We welcome contributions! Please refer to our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

---