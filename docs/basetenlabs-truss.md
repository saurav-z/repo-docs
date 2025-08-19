# Truss: The Simplest Way to Deploy and Serve Your Machine Learning Models

**[Truss](https://github.com/basetenlabs/truss) simplifies the process of deploying and serving your AI/ML models in production, offering a streamlined experience for developers.**

## Key Features for Effortless Model Deployment:

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Rapid Development Cycle:** Benefit from a fast feedback loop with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   **Universal Framework Support:** Truss seamlessly supports models built with all major Python frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Easily deploy your models to Baseten (with more remotes coming soon, starting with AWS SageMaker) with a single command.

## Getting Started with Truss

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart Example: Text Classification

Here's how to package and deploy a simple text classification model using Truss:

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Follow the prompts to name your Truss.

2.  **Navigate to the Truss Directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement Your Model (`model/model.py`):**  Create a `Model` class with `load()` (model loading) and `predict()` (inference) methods.

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

4.  **Define Dependencies (`config.yaml`):**  Specify your model's dependencies, like Transformers and PyTorch:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

5.  **Deploy Your Model:**

    *   Obtain a [Baseten API key](https://app.baseten.co/settings/account/api_keys) (free signup available).
    *   Run `truss push` to deploy to Baseten.

    ```bash
    truss push
    ```

6.  **Test Your Model:**  Invoke your deployed model from the terminal.

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```

    Example Response:

    ```json
    [
      {
        "label": "POSITIVE",
        "score": 0.999873161315918
      }
    ]
    ```

## Example Truss Models

Explore pre-built Trusses for popular models:

*   🦙 [Llama 2](https://github.com/basetenlabs/truss-examples/tree/main/llama) (7B, 13B, and 70B variations)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

And many more examples are available in the [truss-examples](https://github.com/basetenlabs/truss-examples/) repository.

## Contributing

Truss is an open-source project, and we welcome contributions from the ML community! See our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).