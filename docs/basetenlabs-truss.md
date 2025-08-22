# Truss: The Easiest Way to Deploy & Serve Your AI/ML Models

**Quickly and reliably deploy your machine learning models with Truss, a powerful and flexible framework.** ([Original Repository](https://github.com/basetenlabs/truss))

## Key Features of Truss

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Rapid Development Loop:** Leverage a live reload server for fast feedback and skip complex Docker and Kubernetes configurations.
*   **Broad Framework Support:** Truss seamlessly integrates with all major Python-based machine learning frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models with ease using the Baseten platform, with support for other platforms like AWS SageMaker coming soon.
*   **Extensive Examples:** Explore ready-to-use Trusses for popular models like Llama 2, Stable Diffusion XL, and Whisper, along with many more examples to get you started quickly.

## Getting Started with Truss

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Choose a name for your Truss, such as "Text classification."

2.  **Navigate to your new directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement your model (`model/model.py`):**
    Define a `Model` class with `load()` and `predict()` methods.

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

4.  **Define model dependencies (`config.yaml`):**

    Specify your model's dependencies in the `config.yaml` file. Example:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment

Truss is designed for easy deployment, particularly with Baseten.

### Deploying with Baseten

1.  **Get a Baseten API Key:**
    Sign up for a Baseten account at [https://app.baseten.co/signup/](https://app.baseten.co/signup/) and obtain your API key from [https://app.baseten.co/settings/account/api_keys](https://app.baseten.co/settings/account/api_keys).

2.  **Deploy your model:**

    ```bash
    truss push
    ```

    Monitor the deployment progress on your [Baseten model dashboard](https://app.baseten.co/models/).

### Invoking Your Deployed Model

After deployment, test your model:

```bash
truss predict -d '"Truss is awesome!"'
```

Sample Response:

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