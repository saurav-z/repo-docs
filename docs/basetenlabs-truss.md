# Truss: The Easiest Way to Deploy and Serve Your ML Models

**Truss simplifies the process of deploying and managing your machine learning models in production, making it easy to get your models live quickly.**  [Explore the original repository](https://github.com/basetenlabs/truss).

## Key Features of Truss

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Fast Development Loop:** Benefit from a live reload server for rapid iteration and skip complex Docker and Kubernetes configurations.
*   **Broad Framework Support:** Truss seamlessly supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Production-Ready Deployment:** Easily deploy your models to Baseten and soon to other platforms like AWS SageMaker.

## Example Models Available as Trusses

*   🦙 [Llama 2 7B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) ([13B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-13b-chat)) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-70b-chat))
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

and [dozens more examples](https://github.com/basetenlabs/truss-examples/).

## Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

## Quickstart: Deploy a Text Classification Model

This quickstart will guide you through deploying a text classification pipeline from the open-source [`transformers` package](https://github.com/huggingface/transformers).

### Create a Truss

Create a new Truss project using the following command:

```bash
truss init text-classification
```

Follow the prompts, such as naming your Truss `Text classification`.

Navigate into the newly created directory:

```bash
cd text-classification
```

### Implement the Model

The core of a Truss lies within two essential files: `model/model.py` and `config.yaml`.  The `model/model.py` file defines a `Model` class, providing an interface between your ML model and the serving environment.

Implement two key methods within the `Model` class:

*   `load()`: Loads the model when the server starts or is updated.
*   `predict()`: Handles model inference for each request.

Here's an example `model/model.py`:

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

### Configure Dependencies

The `config.yaml` file is crucial for configuring the model serving environment. Specify required dependencies in the `requirements` section.

In `config.yaml`, modify the `requirements` section as follows:

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

## Deployment

Truss is designed for easy deployment, currently supporting Baseten, and soon to be available on other platforms.

### Get a Baseten API Key

To deploy to Baseten, you'll need a [Baseten API key](https://app.baseten.co/settings/account/api_keys).  If you don't have an account, you can [sign up for free](https://app.baseten.co/signup/).

### Deploy with `truss push`

Deploy your model to Baseten using the following command and providing your API key when prompted:

```bash
truss push
```

Monitor the deployment progress on [your Baseten model dashboard](https://app.baseten.co/models/).

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

## Contributing

Truss is backed by Baseten and built with contributions from the community.  We welcome contributions; please consult our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

Special thanks to [Stephan Auerhahn](https://github.com/palp) @ [stability.ai](https://stability.ai/) and [Daniel Sarfati](https://github.com/dsarfati) @ [Salad Technologies](https://salad.com/) for their contributions.