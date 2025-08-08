# Truss: Deploy and Serve Your AI/ML Models with Ease

**Quickly and effortlessly deploy your machine learning models to production using Truss, a powerful and versatile model serving framework.**

[View the original repository on GitHub](https://github.com/basetenlabs/truss)

Truss simplifies the entire model deployment process, allowing you to focus on building and iterating on your AI/ML models. 

## Key Features of Truss

*   ✅ **Write Once, Run Anywhere:** Package your model code, weights, and dependencies into a self-contained unit that behaves consistently across development and production environments.
*   ✅ **Rapid Development Cycle:** Benefit from a fast feedback loop with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   ✅ **Framework Agnostic:** Seamlessly support models built with any Python framework, including popular choices like `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.
*   ✅ **Simplified Deployment:** Deploy your models to production with a single command, streamlining the transition from development to live serving.
*   ✅ **Scalable Infrastructure:** Leverage robust infrastructure provided by Baseten to handle the demands of production workloads.

## Get Started with Truss

### Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

### Quickstart: Deploy a Text Classification Model

Here's how to deploy a text classification model using the popular `transformers` library:

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```

    Give your Truss a name when prompted.

2.  **Navigate to your Truss directory:**

    ```bash
    cd text-classification
    ```

3.  **Implement the Model (`model/model.py`):**  Create a `Model` class that defines `load()` (for model loading) and `predict()` (for inference).

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

4.  **Specify Dependencies (`config.yaml`):**  Add the necessary dependencies in `config.yaml`.

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

### Deployment to Baseten

1.  **Get a Baseten API Key:** Sign up for a free account and obtain your API key from the [Baseten dashboard](https://app.baseten.co/settings/account/api_keys).
2.  **Deploy your model:**

    ```bash
    truss push
    ```

3.  **Invoke the model:**

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```

    **Example Response:**

    ```json
    [
      {
        "label": "POSITIVE",
        "score": 0.999873161315918
      }
    ]
    ```

## Example Trusses

*   🦙 [Llama 2 7B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) ([13B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-13b-chat)) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-70b-chat))
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

And [dozens more examples](https://github.com/basetenlabs/truss-examples/).

## Contribute

We welcome contributions! Please see our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).