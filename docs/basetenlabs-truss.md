# Truss: The Fastest Way to Deploy and Serve Your Machine Learning Models

**[Original Repository](https://github.com/basetenlabs/truss)**

Truss simplifies the process of deploying and serving your AI/ML models, getting them into production with unprecedented speed and ease.

## Key Features of Truss

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a single, portable unit that behaves the same in development and production environments.
*   **Rapid Development Cycle:** Benefit from a fast feedback loop with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   **Broad Framework Support:** Truss seamlessly supports models created using any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.

## Examples

Truss has examples for popular models, including:

*   🦙 [Llama 2 7B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) ([13B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-13b-chat)) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-70b-chat))
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

and [dozens more examples](https://github.com/basetenlabs/truss-examples/).

## Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

## Quickstart: Deploying a Text Classification Model

This quickstart guides you through deploying a text classification pipeline using the `transformers` library.

### 1. Create a Truss

Initialize a new Truss project:

```bash
truss init text-classification
```

Give your Truss a name, like `Text classification`.

Navigate into the newly created directory:

```bash
cd text-classification
```

### 2. Implement Your Model (`model/model.py`)

The core of your Truss is the `model/model.py` file, where you define a `Model` class with `load()` and `predict()` methods:

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

### 3. Configure Dependencies (`config.yaml`)

Specify your model's dependencies in `config.yaml`:

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

## Deployment

Truss is maintained by [Baseten](https://baseten.co), which provides infrastructure for running ML models in production. We'll use Baseten as the remote host for your model.

Other remotes are coming soon, starting with AWS SageMaker.

### Get an API Key

Obtain a [Baseten API key](https://app.baseten.co/settings/account/api_keys). Sign up for an account [here](https://app.baseten.co/signup/) to get started.

### Deploy Your Model

Deploy your model using:

```bash
truss push
```

Monitor your model deployment from [your model dashboard on Baseten](https://app.baseten.co/models/).

### Invoke the Model

After deployment, test your model from the terminal:

**Invocation**

```bash
truss predict -d '"Truss is awesome!"'
```

**Response**

```json
[
  {
    "label": "POSITIVE",
    "score": 0.999873161315918
  }
]
```

## Contributions

Truss is a community-driven project backed by Baseten, with contributions from ML engineers worldwide. We welcome contributions – see our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).