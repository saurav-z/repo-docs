# Truss: The Simplest Way to Deploy & Serve Your AI/ML Models

**Truss** is an open-source framework that simplifies the process of packaging, serving, and deploying your machine learning models, regardless of the framework you use.  Learn more and contribute on the [original repo](https://github.com/basetenlabs/truss).

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features of Truss

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Fast Development Loop:** Benefit from a live reload server, enabling rapid feedback during model implementation and avoiding complex Docker/Kubernetes configurations.
*   **Broad Framework Support:**  Truss seamlessly supports models built with any Python framework, including Transformers, diffusers, PyTorch, TensorFlow, TensorRT, and Triton.
*   **Easy Deployment:**  Deploy your models to Baseten with a single command, with more remote options coming soon, starting with AWS SageMaker.
*   **Simplified Configuration:**  Configure your model serving environment with a straightforward `config.yaml` file.

## Example Models

Truss supports a wide variety of models, including:

*   🦙 [Llama 2](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) (7B, 13B, 70B)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

And [dozens more examples](https://github.com/basetenlabs/truss-examples/).

## Getting Started

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification Example

This quickstart guides you through packaging a text classification pipeline using the Transformers library.

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Choose a name, like `Text classification`. Then navigate to the newly created directory: `cd text-classification`
2.  **Implement the Model (`model/model.py`):** Define a `Model` class with `load()` and `predict()` methods.

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
3.  **Add Model Dependencies (`config.yaml`):** Specify required packages, like `torch` and `transformers`:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

### Deployment

1.  **Get a Baseten API Key:** Sign up for a [Baseten account](https://app.baseten.co/signup/) and obtain your API key from [your account settings](https://app.baseten.co/settings/account/api_keys).
2.  **Deploy with `truss push`:**

    ```bash
    truss push
    ```
3.  **Invoke the Model:**  After deployment, test your model.

    **Invocation:**

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

Truss is a community-driven project.  Contributions are welcome; please refer to our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

## Truss Contributors

Truss is backed by Baseten and built in collaboration with ML engineers worldwide. Special thanks to [Stephan Auerhahn](https://github.com/palp) @ [stability.ai](https://stability.ai/) and [Daniel Sarfati](https://github.com/dsarfati) @ [Salad Technologies](https://salad.com/) for their contributions.