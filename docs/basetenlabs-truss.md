# Truss: The Easiest Way to Deploy and Serve Your Machine Learning Models

**(Original repository: [https://github.com/basetenlabs/truss](https://github.com/basetenlabs/truss))**

Truss simplifies the process of deploying and serving your AI/ML models in production, enabling you to go from development to deployment faster than ever.

**Key Features:**

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies for consistent behavior across development and production environments.
*   **Rapid Development Cycle:** Leverage a live reload server for quick feedback and skip complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Truss supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Easy Deployment:** Seamlessly deploy your models to Baseten, with more platforms coming soon.

**Explore Truss Examples:**

See pre-built Trusses for popular models to get started quickly:

*   🦙 [Llama 2 7B, 13B, and 70B](https://github.com/basetenlabs/truss-examples/tree/main/llama)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

... and [dozens more examples](https://github.com/basetenlabs/truss-examples/).

**Getting Started**

**Installation:**

Install Truss using pip:

```bash
pip install --upgrade truss
```

**Quickstart: Text Classification Example**

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Follow the prompts to name your Truss.

2.  **Navigate to the directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement your Model:**

    Edit `model/model.py` to define a `Model` class with `load()` and `predict()` methods:

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

4.  **Add Model Dependencies:**

    In `config.yaml`, update the `requirements` section:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

**Deployment**

Truss is currently supported by Baseten.

1.  **Get a Baseten API Key:**
    Sign up for a Baseten account at [https://app.baseten.co/signup/](https://app.baseten.co/signup/) and obtain your API key from [https://app.baseten.co/settings/account/api_keys](https://app.baseten.co/settings/account/api_keys).

2.  **Deploy your Model:**

    ```bash
    truss push
    ```

    Monitor your model's deployment on [your Baseten dashboard](https://app.baseten.co/models/).

3.  **Invoke Your Model:**

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

**Contributing**

Truss is a community-driven project. We welcome contributions! Please refer to our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

**Acknowledgements**

Special thanks to [Stephan Auerhahn](https://github.com/palp) @ [stability.ai](https://stability.ai/) and [Daniel Sarfati](https://github.com/dsarfati) @ [Salad Technologies](https://salad.com/) for their contributions.