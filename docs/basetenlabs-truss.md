# Truss: The Easiest Way to Deploy and Serve Your AI/ML Models

[Truss](https://github.com/basetenlabs/truss) simplifies the process of deploying and serving your machine learning models, making it easier to get your AI projects into production.

## Key Features:

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Fast Development Loop:** Iterate quickly with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Supports models built with any Python framework, including Transformers, PyTorch, TensorFlow, TensorRT, and Triton.
*   **Simple Deployment:** Easily deploy your models to Baseten for production. AWS SageMaker and other platforms coming soon.
*   **Example Ready**: Includes pre-made Trusses of popular models such as Llama 2, Stable Diffusion XL and Whisper.

## Get Started Quickly

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Create Your First Truss

1.  **Initialize a Truss:**

    ```bash
    truss init text-classification
    ```

2.  **Navigate to the New Directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement Your Model (model/model.py):**
    Define a `Model` class with `load()` (loads the model) and `predict()` (handles inference) functions. Example:

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

4.  **Add Dependencies (config.yaml):**
    Specify your model's dependencies in `config.yaml`. Example:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment with Baseten

1.  **Get a Baseten API Key:**  Sign up for a Baseten account and get your API key from [Baseten](https://app.baseten.co/settings/account/api_keys).

2.  **Deploy Your Model:**

    ```bash
    truss push
    ```

3.  **Monitor Deployment:** Track your model's deployment on your [Baseten model dashboard](https://app.baseten.co/models/).

4.  **Invoke the Model:**

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

We welcome contributions! See our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).