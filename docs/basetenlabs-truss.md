# Truss: The Simplest Way to Deploy and Serve Your AI/ML Models

**Easily deploy and manage your machine learning models in production with Truss, a powerful and flexible framework.** ([View on GitHub](https://github.com/basetenlabs/truss))

## Key Features

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a consistent model server that behaves the same in development and production environments.
*   **Accelerated Development:** Iterate quickly with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   **Comprehensive Framework Support:** Truss seamlessly integrates with all major Python ML frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and Triton.

## Get Started with Truss

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Deploy a Text Classification Model

1.  **Initialize a Truss:**

    ```bash
    truss init text-classification
    ```

2.  **Navigate to your new directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement Your Model (`model/model.py`):**  Define your model's `load()` and `predict()` functions within the `Model` class.  Here's an example using a Hugging Face text classification pipeline:

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

4.  **Configure Dependencies (`config.yaml`):**  Specify your model's dependencies, such as `transformers` and `torch`:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

### Deployment

Truss is designed for easy deployment.  Here's how to deploy your model using Baseten (with other remotes like AWS SageMaker coming soon):

1.  **Get a Baseten API Key:**  Sign up for a [Baseten account](https://app.baseten.co/signup/) and obtain your API key from [your account settings](https://app.baseten.co/settings/account/api_keys).

2.  **Deploy Your Model:**

    ```bash
    truss push
    ```

    Follow the prompts to enter your Baseten API key.  Monitor your deployment from your [Baseten model dashboard](https://app.baseten.co/models/).

3.  **Invoke Your Model:**  Once deployed, test your model via the terminal:

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```

    You'll receive a JSON response with the model's output.

### Examples

Truss supports a wide range of models. Here are some examples:

*   🦙 [Llama 2](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) (7B, 13B, 70B)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

Find [dozens more examples](https://github.com/basetenlabs/truss-examples/) to get started.

## Contributing

Truss is an open-source project, and we welcome contributions! Please see our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md) for more information.