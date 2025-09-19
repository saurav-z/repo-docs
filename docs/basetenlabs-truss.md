# Truss: The Simplest Way to Deploy and Serve Your AI/ML Models

**(Original Repo: [https://github.com/basetenlabs/truss](https://github.com/basetenlabs/truss))**

Truss is a powerful and user-friendly tool designed to simplify the process of deploying and serving your machine learning models, accelerating your path to production.

## Key Features of Truss:

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Rapid Development Loop:** Benefit from a live reload server for fast feedback and eliminate complex Docker/Kubernetes configurations.
*   **Framework Agnostic:** Supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Easy Deployment:** Seamlessly deploy your models to Baseten (with AWS SageMaker support coming soon).
*   **Simplified Packaging:** Easily package your model with a `model.py` file for inference logic and a `config.yaml` file for configuration.

## Example Models with Truss

Explore pre-built Truss examples for popular models:

*   🦙 [Llama 2](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) (7B, 13B, and 70B variants)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

...and [dozens more examples](https://github.com/basetenlabs/truss-examples/).

## Getting Started with Truss

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Deploy a Text Classification Model

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

4.  **Add Dependencies (`config.yaml`):**

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

5.  **Deployment:**

    *   Get a [Baseten API key](https://app.baseten.co/settings/account/api_keys).  [Sign up](https://app.baseten.co/signup/) if you don't have one.
    *   Run `truss push`.
    *   Monitor deployment on your [Baseten model dashboard](https://app.baseten.co/models/).

6.  **Invocation:**

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

Truss is an open-source project, and we welcome contributions!  See our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

**Thanks to our Contributors!**
Special thanks to Stephan Auerhahn (@palp) from stability.ai and Daniel Sarfati (@dsarfati) from Salad Technologies for their contributions.