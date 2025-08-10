# Truss: Productionize Your Machine Learning Models with Ease

**Truss simplifies the process of deploying and serving your AI/ML models, allowing you to go from development to production quickly and efficiently.**  Explore the [Truss](https://github.com/basetenlabs/truss) repository for more details.

## Key Features

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies in a standardized format that works consistently across development and production environments.
*   **Rapid Development Loop:** Benefit from a live reload server, enabling faster feedback and iterative development without the need for complex Docker or Kubernetes configurations.
*   **Framework Agnostic:** Truss supports a wide range of Python frameworks, including popular choices like `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`, ensuring compatibility with your preferred tools.
*   **Simplified Deployment:** Deploy your models to Baseten with a single command. Additional deployment options, such as AWS SageMaker, are planned for future releases.
*   **Pre-built Examples:** Get started quickly with example Trusses for popular models like Llama 2, Stable Diffusion XL, and Whisper.

## Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

## Quickstart Guide

This quickstart demonstrates how to package a text classification pipeline from the `transformers` package.

### Create a Truss

Create a new Truss project with the following command:

```bash
truss init text-classification
```

Follow the prompts to name your project.

### Implement the Model

1.  Navigate to the project directory:

    ```bash
    cd text-classification
    ```
2.  Open `model/model.py` and define a `Model` class with `load()` and `predict()` methods:

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

### Add Model Dependencies

Edit the `config.yaml` file and specify the required dependencies, like Transformers and PyTorch:

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

## Deployment

Deploy your Truss model to Baseten for production use.

### Get an API Key

Obtain a [Baseten API key](https://app.baseten.co/settings/account/api_keys). Sign up for a free account if you don't have one.

### Run `truss push`

Deploy your model to Baseten:

```bash
truss push
```

Monitor the deployment progress from [your Baseten model dashboard](https://app.baseten.co/models/).

### Invoke the Model

Once deployed, test your model using the command:

```bash
truss predict -d '"Truss is awesome!"'
```

## Truss Contributors

Truss is supported by Baseten and developed in collaboration with the ML community.