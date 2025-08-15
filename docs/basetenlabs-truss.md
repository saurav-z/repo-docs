# Truss: Effortless Deployment for Your AI/ML Models

**Quickly and easily deploy your machine learning models to production with Truss, a powerful and flexible solution.** ([Original Repository](https://github.com/basetenlabs/truss))

## Key Features:

*   **Write Once, Run Anywhere:** Package and test your model code, weights, and dependencies seamlessly, ensuring consistent behavior from development to production.
*   **Fast Developer Loop:** Benefit from a live reload server, accelerating your development cycle and eliminating the need for complex Docker/Kubernetes configurations.
*   **Comprehensive Framework Support:** Truss works with all major Python frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   **Simplified Deployment:** Deploy your models with a single command using `truss push`.
*   **Production-Ready:** Easily deploy to Baseten or AWS SageMaker with more cloud providers coming soon.

## Get Started with Truss

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart Guide: Deploy a Text Classification Model

Here's a quick example to get you started with a text classification pipeline from the `transformers` package:

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Give your Truss a name when prompted (e.g., "Text classification").

2.  **Navigate to your new directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement the Model:**

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

    Edit `config.yaml` to include the necessary dependencies.  Replace the `requirements:` section with:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

5.  **Deployment (using Baseten):**

    *   **Get a Baseten API key:** Sign up for a [Baseten account](https://app.baseten.co/signup/) and obtain an API key from your account settings ([Baseten API key](https://app.baseten.co/settings/account/api_keys)).
    *   **Run `truss push`:**  Deploy your model using the following command and provide your Baseten API key when prompted:

        ```bash
        truss push
        ```

    *   **Monitor Deployment:** Track your model's deployment progress on your [Baseten model dashboard](https://app.baseten.co/models/).

6.  **Invoke the Model:** After deployment, test your model from the terminal.

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```

    **Expected Response:**

    ```json
    [
      {
        "label": "POSITIVE",
        "score": 0.999873161315918
      }
    ]
    ```

## Truss Examples

Explore pre-built Trusses for popular models:

*   🦙 [Llama 2 7B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) ([13B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-13b-chat)) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-70b-chat))
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

And many more examples can be found [here](https://github.com/basetenlabs/truss-examples/).

## Contributing

We welcome contributions!  Please review our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).