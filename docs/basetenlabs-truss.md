# Truss: The Easiest Way to Deploy and Serve Your Machine Learning Models

**Quickly and effortlessly deploy your AI/ML models to production with Truss, the open-source framework designed for simplicity and scalability.** ([View the original repo](https://github.com/basetenlabs/truss))

## Key Features of Truss:

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Fast Developer Loop:** Iterate rapidly with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   **Broad Framework Support:** Truss seamlessly supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models with a single command using `truss push`, streamlining the deployment process.
*   **Pre-built Examples:** Get started quickly with examples for popular models like Llama 2, Stable Diffusion XL, and Whisper.

## Get Started with Truss

### Installation

Install the Truss package using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Deploy a Text Classification Model

Follow these steps to quickly deploy a text classification model using the `transformers` library:

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

4.  **Add Dependencies (`config.yaml`):**

    Update the `requirements` section in `config.yaml`:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

5.  **Deployment:**

    *   Obtain a [Baseten API key](https://app.baseten.co/settings/account/api_keys). Sign up for a free account if you don't have one.
    *   Run `truss push` to deploy your model to Baseten.
    *   Monitor deployment on the [Baseten dashboard](https://app.baseten.co/models/).

6.  **Invocation:**
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

## Contribute

We welcome contributions! See our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).