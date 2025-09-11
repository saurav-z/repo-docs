# Truss: The Easiest Way to Deploy and Serve Your AI/ML Models

**[Explore the original repo](https://github.com/basetenlabs/truss)**

Truss simplifies the process of deploying and serving your machine learning models, making it easier than ever to get your AI projects into production.

## Key Features:

*   **Write Once, Run Anywhere:** Package your model code, dependencies, and weights for consistent behavior across development and production environments.
*   **Rapid Development Cycle:** Utilize a live reload server for quick feedback, eliminating the need for Docker and Kubernetes configurations.
*   **Comprehensive Framework Support:** Truss seamlessly integrates with all major Python ML frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models with ease using the Baseten platform or, soon, AWS SageMaker.
*   **Ready-to-Use Examples:** Get started quickly with pre-built Trusses for popular models like Llama 2, Stable Diffusion XL, and Whisper.

## Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

## Quickstart: Deploy a Text Classification Model

Here's a simplified example of how to deploy a text classification model using Truss:

1.  **Initialize a Truss:**

    ```bash
    truss init text-classification
    ```

2.  **Navigate to the directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement Your Model (`model/model.py`):**

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

4.  **Define Dependencies (`config.yaml`):**

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

5.  **Deploy to Baseten:**
    *   Get a [Baseten API key](https://app.baseten.co/settings/account/api_keys).  If you don't have a Baseten account, [sign up for an account](https://app.baseten.co/signup/).
    *   Run `truss push` to deploy your model.
    *   Monitor deployment on your [model dashboard on Baseten](https://app.baseten.co/models/).

6.  **Invoke Your Model:**

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```

## Community and Contributions

Truss is backed by Baseten and built in collaboration with ML engineers worldwide.  We welcome contributions; see our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).