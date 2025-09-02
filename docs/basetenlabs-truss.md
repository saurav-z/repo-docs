# Truss: The Easiest Way to Deploy and Serve Your AI/ML Models

**Truss** simplifies the process of deploying and serving your machine learning models in production, providing a seamless experience from development to deployment. Get started with Truss on [GitHub](https://github.com/basetenlabs/truss).

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Fast Developer Loop:** Benefit from a live reload server for rapid feedback and eliminate the need for extensive Docker and Kubernetes configurations.
*   **Comprehensive Framework Support:** Truss seamlessly integrates with all major Python ML frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models with a single command using Baseten, with support for other platforms like AWS SageMaker coming soon.
*   **Ready-to-Use Examples:** Explore pre-built Trusses for popular models like Llama 2, Stable Diffusion XL, and Whisper, and many more.

## Getting Started

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification Example

Here's how to quickly package and deploy a text classification model using Truss:

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
    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment

Truss is designed for easy deployment, currently supporting [Baseten](https://baseten.co) with more platforms coming soon.

1.  **Get a Baseten API Key:** Sign up for a [Baseten](https://app.baseten.co/signup/) account and obtain an API key from your [account settings](https://app.baseten.co/settings/account/api_keys).
2.  **Deploy with `truss push`:**
    ```bash
    truss push
    ```
3.  **Monitor Deployment:** Track your model's deployment progress on your [Baseten model dashboard](https://app.baseten.co/models/).
4.  **Invoke the Model:** After deployment, test your model from the terminal:

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```
    ```json
    [
      {
        "label": "POSITIVE",
        "score": 0.999873161315918
      }
    ]
    ```

## Contributing

Truss is a community-driven project. We welcome contributions! Please refer to our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md) for details.

## Acknowledgements

Special thanks to [Stephan Auerhahn](https://github.com/palp) @ [stability.ai](https://stability.ai/) and [Daniel Sarfati](https://github.com/dsarfati) @ [Salad Technologies](https://salad.com/) for their contributions.