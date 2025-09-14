# Truss: The Simplest Way to Serve Your AI/ML Models in Production

Easily package, deploy, and manage your machine learning models with **Truss**, a powerful and versatile framework designed for production. Learn more on the [original repo](https://github.com/basetenlabs/truss).

## Key Features:

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Fast Development Loop:** Iterate quickly with a live reload server, streamlining your development process without complex Docker or Kubernetes configurations.
*   **Framework Agnostic:** Supports a wide range of Python frameworks including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Seamlessly deploy your models to Baseten (with other platforms like AWS SageMaker coming soon) with a single command.
*   **Pre-built Examples:** Get started quickly with pre-configured Trusses for popular models like Llama 2, Stable Diffusion XL, and Whisper, plus dozens more in the [examples repository](https://github.com/basetenlabs/truss-examples/).

## Getting Started

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification Example

Here's how to package and deploy a text classification model using Truss:

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Provide a name for your Truss, such as "Text classification".

2.  **Navigate to the directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement the Model (`model/model.py`):**

    Create a `Model` class within `model/model.py` with `load()` and `predict()` methods. Here's an example:

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

4.  **Add Model Dependencies (`config.yaml`):**

    Edit `config.yaml` to include your model's dependencies. For the text classification example, add the following under `requirements`:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment with Baseten

1.  **Get a Baseten API Key:**

    Create an account or log in to [Baseten](https://app.baseten.co/signup/) and obtain your API key from your account settings.

2.  **Run `truss push`:**

    Deploy your model to Baseten:

    ```bash
    truss push
    ```

    Follow the prompts, providing your Baseten API key when requested. Monitor your model deployment on your [Baseten model dashboard](https://app.baseten.co/models/).

3.  **Invoke the Model:**

    Once deployed, you can test your model from the terminal:

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

We welcome contributions! Please refer to our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).