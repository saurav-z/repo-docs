# Truss: The Easiest Way to Deploy Your Machine Learning Models

**Truss** simplifies the process of deploying your machine learning models to production, offering a streamlined solution for developers of all levels. ([View the original repository](https://github.com/basetenlabs/truss))

## Key Features

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies, ensuring consistent behavior across development and production environments.
*   **Rapid Development Loop:** Iterate quickly with a live reload server, reducing the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Supports all major Python machine learning frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models to Baseten with a simple `truss push` command.  AWS SageMaker integration is coming soon!
*   **Pre-built Examples:** Get started quickly with ready-to-use examples for popular models like Llama 2, Stable Diffusion XL, and Whisper.

## Get Started with Truss

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

    Choose a name for your Truss (e.g., "Text classification").

2.  **Navigate to the Directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement Your Model (`model/model.py`):**

    Create a `Model` class with `load()` and `predict()` functions:

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

    In `config.yaml`, specify the required dependencies:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

### Deployment with Baseten

1.  **Get a Baseten API Key:** [Sign up for a Baseten account](https://app.baseten.co/signup/) and obtain your API key from your account settings.

2.  **Deploy Your Model:**

    ```bash
    truss push
    ```

    Follow the prompts to deploy to Baseten.  Monitor your model deployment from [your model dashboard on Baseten](https://app.baseten.co/models/).

3.  **Invoke Your Model:**

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```

    You should receive a JSON response with the predicted sentiment.

## Contributing

We welcome contributions!  Please review our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).