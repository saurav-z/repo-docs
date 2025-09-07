# Truss: The Simplest Way to Deploy Your AI/ML Models

**[Go to the original repo](https://github.com/basetenlabs/truss)**

Truss simplifies the process of serving your AI/ML models in production, offering a streamlined solution for developers.

## Key Features:

*   **Write Once, Run Anywhere:** Package your model code, weights, and dependencies with a consistent model server environment for development and production.
*   **Rapid Development:** Benefit from a fast feedback loop with a live reload server, eliminating the need for complex Docker and Kubernetes configurations.
*   **Framework Agnostic:** Supports models built with any Python framework, including `transformers`, `diffusers`, `PyTorch`, `TensorFlow`, `TensorRT`, and `Triton`.

## Examples & Supported Models:

Explore pre-built Trusses for popular models:

*   🦙 [Llama 2](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) (7B, 13B, and 70B variants)
*   🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
*   🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

and many [more examples](https://github.com/basetenlabs/truss-examples/).

## Installation

Install Truss using pip:

```bash
pip install --upgrade truss
```

## Quickstart

Here's how to quickly deploy a text classification pipeline:

### Create a Truss

Run the following command to initialize a new Truss:

```bash
truss init text-classification
```

Provide a name for your Truss, such as "Text classification."

Navigate into the new directory:

```bash
cd text-classification
```

### Implement the Model

Key files in a Truss include:

*   `model/model.py`:  Define a `Model` class with `load()` and `predict()` methods for model interaction.
*   `config.yaml`: Configure the model environment, including dependencies like Transformers and PyTorch.

## Deployment

Truss is designed to be deployed easily using Baseten:

### Get an API Key

Obtain a [Baseten API key](https://app.baseten.co/settings/account/api_keys). Sign up for a free account if you don't have one.

### Run `truss push`

Deploy your model with:

```bash
truss push
```

Monitor the deployment progress via [your Baseten model dashboard](https://app.baseten.co/models/).

### Invoke the Model

After deployment, test your model from the terminal:

**Invocation**

```bash
truss predict -d '"Truss is awesome!"'
```

**Response (example)**

```json
[
  {
    "label": "POSITIVE",
    "score": 0.999873161315918
  }
]
```

## Contributing

Truss is a community-driven project. We welcome contributions, following our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

## Acknowledgements

Special thanks to [Stephan Auerhahn](https://github.com/palp) @ [stability.ai](https://stability.ai/) and [Daniel Sarfati](https://github.com/dsarfati) @ [Salad Technologies](https://salad.com/) for their contributions.