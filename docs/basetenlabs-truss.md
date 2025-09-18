# Truss: The Simplest Way to Deploy and Serve Your AI/ML Models

**Truss** simplifies the process of deploying and serving your machine learning models, allowing you to focus on building and innovating. Learn more on the [original repo](https://github.com/basetenlabs/truss).

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features of Truss

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies, and serve them consistently across development and production environments.
*   **Fast Development Loop:** Benefit from a live reload server for rapid feedback during model implementation, eliminating the need for complex Docker and Kubernetes configurations.
*   **Comprehensive Framework Support:** Truss works seamlessly with all major Python-based ML frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models easily with a focus on integration with Baseten, and with future integrations for AWS SageMaker.

## Ready-to-Use Examples

Explore pre-built Trusses for popular models:

*   🦙 [Llama 2 (7B, 13B, 70B)](https://github.com/basetenlabs/truss-examples/tree/main/llama)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

And [dozens more examples](https://github.com/basetenlabs/truss-examples/) to get you started quickly.

## Getting Started

### Installation

Install the Truss package with pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification Example

1.  **Create a Truss:**
    ```bash
    truss init text-classification
    ```
    Name your Truss (e.g., "Text classification").

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

## Deployment

Truss is maintained by Baseten.  For deployment instructions, see the original repo.

## Community and Contributions

Truss is backed by Baseten and welcomes contributions. See the [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).