# Truss: The Simplest Way to Deploy and Serve Your AI/ML Models

**[Truss](https://github.com/basetenlabs/truss) simplifies the process of taking your machine learning models from development to production, enabling faster and more reliable deployments.**

<br/>

Truss is a powerful tool for deploying and serving AI/ML models. It allows you to package your model code, weights, and dependencies into a single, portable unit, ensuring consistent behavior across different environments. Whether you're working with `transformers`, `PyTorch`, `TensorFlow`, or other frameworks, Truss provides a streamlined solution for model deployment.

**Key Features:**

*   **Write Once, Run Anywhere:** Package your model and dependencies for consistent behavior in development and production.
*   **Fast Development Loop:** Leverage a live reload server for rapid iteration and skip complex Docker/Kubernetes configurations.
*   **Broad Framework Support:** Compatible with all major Python ML frameworks, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and Triton.
*   **Simplified Deployment:** Deploy your models to Baseten or other supported platforms (with AWS SageMaker support coming soon) with a simple command.
*   **Quickstart:** Easy to get started with a text classification pipeline using a straightforward `truss init` command.

<br/>

**Examples of Supported Models:**

*   🦙 [Llama 2 7B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) ([13B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-13b-chat)) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-70b-chat))
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

<br/>

**Installation:**

Install Truss using pip:

```bash
pip install --upgrade truss
```

<br/>

**Quickstart: Deploying a Text Classification Model**

1.  **Create a Truss:**

    ```bash
    truss init text-classification
    ```
2.  **Navigate to the new directory:**

    ```bash
    cd text-classification
    ```
3.  **Implement the Model** Edit `model/model.py` to define your model's loading and prediction logic:

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
4.  **Add Model Dependencies** Specify dependencies in `config.yaml`:

    ```yaml
    requirements:
      - torch==2.0.1
      - transformers==4.30.0
    ```

<br/>

**Deployment:**

1.  **Get a Baseten API Key:** Sign up for a Baseten account [here](https://app.baseten.co/signup/) and obtain your API key from your account settings [here](https://app.baseten.co/settings/account/api_keys).
2.  **Run `truss push`:** Deploy your model to Baseten with the following command, and provide your API key when prompted:

    ```bash
    truss push
    ```

3.  **Monitor and Invoke:** Monitor your deployment on the Baseten platform, then invoke the model using the `truss predict` command:

    ```bash
    truss predict -d '"Truss is awesome!"'
    ```

<br/>

**Contributors:**

Truss is developed by Baseten in collaboration with ML engineers worldwide.  Special thanks to Stephan Auerhahn from stability.ai and Daniel Sarfati from Salad Technologies for their contributions.  We welcome contributions in accordance with our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).