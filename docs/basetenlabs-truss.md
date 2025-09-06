# Truss: The Easiest Way to Deploy and Serve Your AI/ML Models

**Truss** is a powerful tool for simplifying the deployment of your machine learning models, allowing you to focus on building and innovating.  ([See the original repo](https://github.com/basetenlabs/truss))

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/py/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Key Features of Truss:

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a consistent environment that works the same in development and production.
*   **Accelerated Development:** Benefit from a rapid development loop with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Supports all major Python ML frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Easy Deployment:** Seamlessly deploy your models to platforms like Baseten (with more coming soon) for effortless production serving.

## Getting Started with Truss

### Installation

Install Truss easily using pip:

```bash
pip install --upgrade truss
```

## Quickstart: Deploying a Text Classification Model

Here's how to get started with a simple text classification example using the Hugging Face `transformers` library.

### Create a Truss

Use the following command to initialize a new Truss project:

```bash
truss init text-classification
```

When prompted, give your Truss a descriptive name, such as "Text classification." Then, navigate to the newly created directory:

```bash
cd text-classification
```

### Implement the Model (`model/model.py`)

Within your Truss, you'll define a `Model` class. This class acts as the interface between your ML model and the Truss server.  The `load()` method loads your model, and the `predict()` method handles inference.

Here's a complete example for text classification:

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

### Configure Dependencies (`config.yaml`)

The `config.yaml` file allows you to specify your model's dependencies.  For the text classification example, include:

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

### Deploying Your Model

1.  **Get a Baseten API Key:** Create an account or sign in to [Baseten](https://app.baseten.co/) to obtain your API key.
2.  **Deploy with `truss push`:**

    ```bash
    truss push
    ```

    You'll be prompted for your Baseten API key.

3.  **Monitor Deployment:** Track your model's deployment progress on your Baseten dashboard ([your model dashboard on Baseten](https://app.baseten.co/models/)).

### Invoke the Model

Once deployed, you can test your model:

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

## Examples

Explore pre-built Trusses for popular models:

*   🦙 [Llama 2 7B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) ([13B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-13b-chat)) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-70b-chat))
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

And many more examples are available in the [Truss examples repository](https://github.com/basetenlabs/truss-examples/).

## Contributing

Contributions are welcome!  Please refer to our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

**Thanks to our contributors:** [Stephan Auerhahn](https://github.com/palp) @ [stability.ai](https://stability.ai/) and [Daniel Sarfati](https://github.com/dsarfati) @ [Salad Technologies](https://salad.com/)