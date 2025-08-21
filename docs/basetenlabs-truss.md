# Truss: Deploy Your AI/ML Models with Ease

**Simplify and accelerate the deployment of your AI/ML models to production with Truss, a powerful and flexible framework.** ([Original Repo](https://github.com/basetenlabs/truss))

## Key Features & Benefits

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Fast Developer Loop:** Enjoy rapid feedback with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   **Broad Framework Support:** Truss seamlessly integrates with all major Python ML frameworks, including Transformers, Diffusers, PyTorch, TensorFlow, TensorRT, and Triton.
*   **Production-Ready Deployment:** Easily deploy your models to Baseten and other platforms (with AWS SageMaker coming soon)
*   **Simple setup with CLI:** Use the `truss init` command for easy Truss creation and the `truss push` to deploy.

## Get Started Quickly

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Text Classification Example

This example demonstrates how to package and deploy a text classification pipeline from the Hugging Face `transformers` library.

**1. Create a Truss:**

```bash
truss init text-classification
```

**2. Navigate to the Truss directory:**

```bash
cd text-classification
```

**3. Implement Your Model (`model/model.py`):**

Define a `Model` class with `load()` (model loading) and `predict()` (inference) methods.

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

**4. Add Dependencies (`config.yaml`):**

Specify required dependencies in `config.yaml`.

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

## Deployment

1.  **Get a Baseten API Key:** Sign up for a free Baseten account and obtain your API key.
2.  **Deploy with `truss push`:**

    ```bash
    truss push
    ```

3.  **Invoke Your Model:** After deployment, test your model.

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

## Examples

Explore existing Trusses for popular models like:

*   🦙 [Llama 2](https://github.com/basetenlabs/truss-examples/tree/main/llama)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

And many more examples in the [Truss Examples Repository](https://github.com/basetenlabs/truss-examples/).

## Contributing

We welcome contributions! See our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).