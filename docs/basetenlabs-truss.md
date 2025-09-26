# Truss: The Easiest Way to Deploy and Serve Your AI/ML Models in Production

**(Original repo: [https://github.com/basetenlabs/truss](https://github.com/basetenlabs/truss))**

Truss simplifies the process of deploying and managing your machine learning models, allowing you to focus on building and iterating.

## Key Features:

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a standardized format that works consistently across development and production environments.
*   **Fast Development Loop:** Benefit from a live reload server for rapid feedback and iteration, eliminating the need for complex Docker or Kubernetes configurations.
*   **Framework Agnostic:** Truss supports models built with all major Python ML frameworks, including Transformers, PyTorch, TensorFlow, TensorRT, and Triton.
*   **Simplified Deployment:** Deploy your models to the cloud with a single command, leveraging Baseten's infrastructure or, soon, AWS SageMaker.

## Example Models (Trusses)

Explore pre-built Truss examples for popular models:

*   🦙 [Llama 2](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) (7B, 13B, 70B)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

Find [dozens more examples](https://github.com/basetenlabs/truss-examples/) to get started.

## Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

## Quickstart: Deploy a Text Classification Model

1.  **Create a Truss:**
    ```bash
    truss init text-classification
    ```
    Give your Truss a name like "Text classification".

2.  **Navigate to the directory:**
    ```bash
    cd text-classification
    ```

3.  **Implement the Model (model/model.py):**
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
    In `config.yaml`, replace the `requirements` list with:
    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

## Deployment to Baseten

1.  **Get a Baseten API Key:**  Create an account and retrieve your API key from [Baseten](https://app.baseten.co/settings/account/api_keys).
2.  **Deploy:**
    ```bash
    truss push
    ```
    Monitor your model deployment on [Baseten](https://app.baseten.co/models/).

3.  **Invoke the Model:**
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

Truss is an open-source project backed by Baseten and built in collaboration with the ML community. Contributions are welcome; refer to our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).