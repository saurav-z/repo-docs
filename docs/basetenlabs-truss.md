# Truss: The Easiest Way to Deploy & Serve Your AI/ML Models

**Effortlessly package, deploy, and serve your machine learning models with Truss, the open-source framework designed for production.**  See the original repo [here](https://github.com/basetenlabs/truss).

## Key Features & Benefits

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a single unit that behaves the same in development and production environments.
*   **Fast Development Loop:**  Get rapid feedback with a live reload server, simplifying debugging and iteration without complex Docker or Kubernetes configurations.
*   **Framework Agnostic:**  Supports all major Python machine learning frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models with ease to platforms like Baseten (and soon, AWS SageMaker) with a single command.
*   **Pre-built Examples:** Get started quickly with Truss examples for popular models such as:
    *   🦙 [Llama 2](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) (7B, 13B, 70B)
    *   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
    *   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)
    *   And [dozens more](https://github.com/basetenlabs/truss-examples/)!

## Getting Started

### Installation

Install the Truss Python package using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification Example

Follow these steps to deploy a text classification model using the Hugging Face `transformers` library.

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Choose a name for your Truss (e.g., "Text classification").
2.  **Navigate to the Directory:**

    ```bash
    cd text-classification
    ```
3.  **Implement the Model (`model/model.py`):**

    Create or modify `model/model.py` to include a `Model` class with `load()` and `predict()` methods:

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

    Edit `config.yaml` to include the necessary dependencies (PyTorch and Transformers):

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment

Truss is designed for seamless deployment to various platforms.  The primary method currently uses Baseten, a platform for production ML models.

1.  **Get a Baseten API Key:**  Sign up for a free Baseten account and obtain your API key from your account settings.
2.  **Push Your Model:**

    ```bash
    truss push
    ```

    The CLI will prompt you for your Baseten API key.

3.  **Monitor Deployment:**  Track the deployment progress via your model dashboard on [Baseten](https://app.baseten.co/models/).
4.  **Invoke the Model:**

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```

    **Example Response:**

    ```json
    [
      {
        "label": "POSITIVE",
        "score": 0.999873161315918
      }
    ]
    ```

## Contributing

Truss is an open-source project backed by Baseten and welcomes contributions from the community.  See our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md) for more information.

**Special Thanks:**  To Stephan Auerhahn (@palp) from [stability.ai](https://stability.ai/) and Daniel Sarfati (@dsarfati) from [Salad Technologies](https://salad.com/) for their contributions.