# Truss: Deploy Your Machine Learning Models Effortlessly

**Truss simplifies the process of deploying and serving your AI/ML models, enabling you to go from development to production with ease.** ([Original Repo](https://github.com/basetenlabs/truss))

## Key Features:

*   **Write Once, Run Anywhere:** Package and test your model code, weights, and dependencies in a consistent environment for both development and production.
*   **Rapid Development:** Leverage a live reload server for quick feedback, eliminating the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Streamlined deployment to Baseten and other platforms, with AWS SageMaker support coming soon.
*   **Pre-built Examples:** Get started quickly with example Trusses for popular models like Llama 2, Stable Diffusion XL, and Whisper.

## Getting Started

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
    Give your Truss a name, such as "Text classification".

2.  **Navigate to the directory:**
    ```bash
    cd text-classification
    ```

3.  **Implement the Model (`model/model.py`):** Create a `Model` class with `load()` and `predict()` methods:

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

4.  **Add Dependencies (`config.yaml`):** Specify your model's dependencies, such as `transformers` and `torch`:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment

Truss seamlessly integrates with Baseten for production deployment.

1.  **Get a Baseten API Key:** Sign up for a Baseten account at [https://app.baseten.co/signup/](https://app.baseten.co/signup/) and obtain your API key from [https://app.baseten.co/settings/account/api_keys](https://app.baseten.co/settings/account/api_keys).

2.  **Deploy with `truss push`:**
    ```bash
    truss push
    ```

3.  **Invoke the Model:** After deployment, test your model:

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

## Contributions

Truss is an open-source project backed by Baseten and built in collaboration with the ML community. Contributions are welcome, please refer to the [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).