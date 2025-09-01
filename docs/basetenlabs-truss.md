# Truss: Deploy and Serve Your AI/ML Models with Ease

**Truss is the easiest way to deploy and serve your machine learning models in production, providing a streamlined experience from development to deployment.**

Learn more about Truss on [GitHub](https://github.com/basetenlabs/truss).

## Key Features

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a self-contained unit that behaves the same in development and production.
*   **Fast Development Cycle:** Utilize a live reload server for rapid feedback and skip complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Supports models built with any Python-based ML framework, including TensorFlow, PyTorch, Transformers, and more.
*   **Simplified Deployment:**  Easily deploy your models to Baseten (with AWS SageMaker coming soon) using a simple command.
*   **Production-Ready:** Designed for scalability and reliability in production environments.

## Getting Started

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification Example

This quickstart demonstrates how to package and deploy a text classification model using the Hugging Face `transformers` library.

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Provide a name when prompted, such as "Text classification."

2.  **Navigate to the New Directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement the Model (`model/model.py`):**

    Create a `Model` class with `load()` (model loading) and `predict()` (inference) methods.

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

    Specify the required libraries (e.g., `transformers`, `torch`) in the `config.yaml` file under the `requirements` section.

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment

### Baseten Deployment

1.  **Get a Baseten API Key:**  Sign up for a Baseten account and obtain your API key.

2.  **Push Your Model:**

    ```bash
    truss push
    ```

    Follow the prompts to deploy your model to Baseten.

3.  **Invoke the Model:**

    After deployment, invoke your model using:

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```

    You will receive a JSON response with the model's predictions.

## Examples

*   🦙 [Llama 2 7B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) ([13B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-13b-chat)) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-70b-chat))
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

and [dozens more examples](https://github.com/basetenlabs/truss-examples/).

## Contributing

We welcome contributions!  See our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).