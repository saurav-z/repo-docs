# Truss: Production-Ready AI/ML Model Serving

**Easily deploy and manage your machine learning models in production with Truss, a powerful and flexible framework.** Learn more and contribute on the original repo: [https://github.com/basetenlabs/truss](https://github.com/basetenlabs/truss)

## Key Features

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Fast Development Loop:** Utilize a live reload server for rapid feedback and iterative development, eliminating the need for complex Docker/Kubernetes configurations.
*   **Broad Framework Support:** Seamlessly integrate models built with any Python framework, including Transformers, Diffusers, PyTorch, TensorFlow, TensorRT, and Triton.
*   **Simplified Deployment:** Easily deploy your models to Baseten and other supported platforms (with AWS SageMaker coming soon).
*   **Comprehensive Examples:** Explore a wide range of pre-built Truss examples for popular models like Llama 2, Stable Diffusion XL, and Whisper, along with many more [examples](https://github.com/basetenlabs/truss-examples/).

## Getting Started

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification

Here's a quick guide to get you started with a text classification model using the popular `transformers` library.

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Follow the prompts to name your Truss (e.g., "Text classification").
2.  **Navigate to your new directory:**

    ```bash
    cd text-classification
    ```
3.  **Implement the Model (`model/model.py`):**  Define a `Model` class with `load()` (for model initialization) and `predict()` (for inference) methods:

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
4.  **Add Model Dependencies (`config.yaml`):** Specify dependencies like `transformers` and `torch` in the `requirements` section:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

### Deployment

1.  **Get a Baseten API Key:**  Sign up for a Baseten account and obtain your API key ([https://app.baseten.co/settings/account/api_keys](https://app.baseten.co/settings/account/api_keys)).
2.  **Push Your Model:**

    ```bash
    truss push
    ```

3.  **Monitor Deployment:** Track your model's deployment progress on the Baseten dashboard ([https://app.baseten.co/models/](https://app.baseten.co/models/)).
4.  **Invoke the Model:**  Once deployed, test your model using the `truss predict` command:

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```

    You should receive a JSON response with the predicted label and confidence score.

## Contributing

We welcome contributions! Please see our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md) for details.