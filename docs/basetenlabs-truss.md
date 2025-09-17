# Truss: The Easiest Way to Deploy Your AI/ML Models

**[View the original repository](https://github.com/basetenlabs/truss)**

Truss simplifies the process of deploying your machine learning models, allowing you to focus on building and innovating, not infrastructure.

## Key Features:

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a consistent environment for seamless deployment across development and production.
*   **Fast Development Loop:** Leverage a live reload server for rapid feedback and iterative development, eliminating the need for complex Docker or Kubernetes configurations.
*   **Framework Agnostic:**  Supports models built with any Python ML framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models with a single command using Baseten, with support for other platforms like AWS SageMaker coming soon.

## Ready-to-Use Examples

Explore pre-built Trusses for popular models:

*   🦙 [Llama 2](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) (7B, 13B, 70B)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

And [dozens more examples](https://github.com/basetenlabs/truss-examples/) to kickstart your projects!

## Getting Started

### Installation

Install Truss easily with pip:

```bash
pip install --upgrade truss
```

### Quickstart: Deploying a Text Classification Model

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Choose a name for your Truss, like "Text classification".
2.  **Navigate to the Truss directory:**

    ```bash
    cd text-classification
    ```
3.  **Implement the Model (model/model.py):**

    Create a `Model` class with `load()` (model initialization) and `predict()` (inference) methods.

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

4.  **Add Dependencies (config.yaml):**

    Specify your model's dependencies in `config.yaml`.  For the text classification example:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment with Baseten

Truss integrates with Baseten for simplified model deployment.

### 1. Get a Baseten API Key

*   Create a free account and get your API key at [Baseten](https://app.baseten.co/signup/).

### 2. Deploy Your Model

```bash
truss push
```

Monitor your model deployment on your [Baseten model dashboard](https://app.baseten.co/models/).

### 3. Invoke Your Model

After deployment, test your model from the terminal:

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

Truss is an open-source project backed by Baseten, built with contributions from the ML community.  We welcome contributions! Please see our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).