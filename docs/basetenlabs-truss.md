# Truss: The Simplest Way to Deploy and Serve Your Machine Learning Models

**Easily package, deploy, and serve your AI/ML models in production with Truss, enabling a smooth transition from development to scalable deployment.** (See the original repo: [https://github.com/basetenlabs/truss](https://github.com/basetenlabs/truss))

## Key Features:

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Rapid Development Loop:** Benefit from a live reload server for quick feedback and eliminate complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models with a single command, leveraging Baseten for production infrastructure, with AWS SageMaker support coming soon.
*   **Easy to Use:** Get started quickly with a simple CLI and intuitive configuration.

## Getting Started

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification Example

This example demonstrates how to package and deploy a text classification model using the `transformers` library.

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

2.  **Navigate to the Directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement the Model (`model/model.py`):**

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
    Replace the `requirements` list in `config.yaml` with the following:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment with Baseten

1.  **Get a Baseten API Key:**  Sign up for a free Baseten account at [https://app.baseten.co/signup/](https://app.baseten.co/signup/) and obtain your API key.
2.  **Push to Baseten:**

    ```bash
    truss push
    ```

    Monitor the deployment on your [Baseten model dashboard](https://app.baseten.co/models/).
3.  **Invoke the Model:**

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

## Example Trusses

*   🦙 [Llama 2 7B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) ([13B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-13b-chat)) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-70b-chat))
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

and [dozens more examples](https://github.com/basetenlabs/truss-examples/).

## Contributing

We welcome contributions! Please see our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).