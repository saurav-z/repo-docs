# Truss: Production-Ready Model Serving Made Simple

**Truss simplifies the deployment of your AI/ML models, enabling you to go from development to production with ease.** ([Original Repo](https://github.com/basetenlabs/truss))

## Key Features:

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a consistent environment for both development and production.
*   **Rapid Development Loop:** Iterate quickly with live reload servers, eliminating the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Easily deploy your models to Baseten, with AWS SageMaker support coming soon.
*   **Pre-built Examples:** Get started quickly with pre-packaged Trusses for popular models like Llama 2, Stable Diffusion XL, and Whisper, and many more.

## Get Started:

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

2.  **Navigate to the directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement Your Model (`model/model.py`):**

    Define your model class with `load()` and `predict()` methods.

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

    Specify your model's requirements:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment: Baseten

1.  **Get a Baseten API Key:**  Sign up for a free account at [https://app.baseten.co/signup/](https://app.baseten.co/signup/) and obtain your API key.

2.  **Deploy Your Model:**

    ```bash
    truss push
    ```

3.  **Monitor Deployment:**  Track your model's progress on the Baseten dashboard: [https://app.baseten.co/models/](https://app.baseten.co/models/)

4.  **Invoke Your Model:**

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

We welcome contributions! Please see our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).