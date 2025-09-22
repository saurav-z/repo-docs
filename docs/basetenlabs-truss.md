# Truss: Deploy AI/ML Models with Ease 🚀

**Truss simplifies the process of serving your machine learning models in production, enabling rapid deployment and scalability.** ([View the original repo](https://github.com/basetenlabs/truss))

## Key Features:

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a consistent environment that behaves the same in development and production.
*   **Fast Developer Loop:** Benefit from a live reload server for immediate feedback, eliminating the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Truss supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Seamlessly deploy your models to Baseten (with AWS SageMaker support coming soon) with a single command.
*   **Pre-built Examples:** Ready-to-use Trusses for popular models like Llama 2, Stable Diffusion XL, and Whisper.

## Getting Started

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Deploy a Text Classification Model

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Name your Truss, e.g., "Text classification".

2.  **Navigate to the directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement the Model (`model/model.py`):**  Define your model's `load()` and `predict()` methods within a `Model` class.

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

4.  **Add Dependencies (`config.yaml`):** Specify your model's requirements.  For example:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment with Baseten

1.  **Get a Baseten API Key:** Sign up at [Baseten](https://app.baseten.co/signup/) and obtain your API key from [your Baseten account settings](https://app.baseten.co/settings/account/api_keys).
2.  **Push Your Model:**

    ```bash
    truss push
    ```

    Monitor deployment on [your Baseten model dashboard](https://app.baseten.co/models/).

3.  **Invoke the Model:** After deployment, test your model.

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```

    Example Response:

    ```json
    [
      {
        "label": "POSITIVE",
        "score": 0.999873161315918
      }
    ]
    ```

## Contribute

We welcome contributions!  See our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).