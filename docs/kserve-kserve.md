# KServe: Production-Ready Model Serving on Kubernetes

**KServe simplifies and accelerates machine learning model deployment on Kubernetes, providing a powerful and scalable solution for serving your AI models.** Explore the original repository on [GitHub](https://github.com/kserve/kserve).

KServe is a Kubernetes Custom Resource Definition (CRD) designed to streamline the deployment and management of machine learning (ML) models, including predictive and generative AI models. This platform addresses production model serving needs by offering user-friendly interfaces for frameworks such as TensorFlow, XGBoost, ScikitLearn, PyTorch, and Hugging Face Transformer/LLM models. It uses standardized data plane protocols.

## Key Features

*   **Simplified Deployment:** Easily deploy and manage ML models on Kubernetes.
*   **Standardized Inference Protocol:** Supports a performant, standardized inference protocol, including OpenAI specifications for generative models.
*   **Serverless Inference:** Enables serverless inference with request-based autoscaling, including scale-to-zero on CPU and GPU.
*   **Advanced Autoscaling:** Built-in autoscaling capabilities optimize resource usage based on demand.
*   **Canary Rollouts:** Facilitates safe and controlled model updates using canary deployments.
*   **ModelMesh Integration:** Integrates with ModelMesh for high scalability, density packing, and dynamic model updates.
*   **Pre/Post-Processing & Explainability:** Provides a simple, pluggable system for inference, pre/post-processing, monitoring, and explainability.
*   **Comprehensive Support:** Supports advanced deployments for canary rollouts, pipelines, and ensembles through InferenceGraph.
*   **Cloud Agnostic:** Works across various cloud platforms.

## Installation

KServe offers several installation options to fit your needs:

*   **Serverless Installation:** Leverages Knative for serverless deployment.
*   **Raw Kubernetes Deployment:** A lightweight installation option.
*   **ModelMesh Installation:** Optional installation for high-scale and dynamic model serving.
*   **Quick Installation:** Install KServe locally for testing and development.
*   **Kubeflow Integration:** Integrates as an important add-on component within Kubeflow.

## Resources

*   **Learn More:** [KServe Website](https://kserve.github.io/website/)
*   **Quickstart Guide:** [Get Started with KServe](https://kserve.github.io/website/docs/getting-started/quickstart-guide)
*   **Create your first InferenceService:** [Get Started with KServe](https://kserve.github.io/website/docs/getting-started/genai-first-isvc)
*   **Roadmap:** [Roadmap](./ROADMAP.md)
*   **InferenceService API Reference:** [API Reference](https://kserve.github.io/website/docs/reference/crd-api)
*   **Developer Guide:** [Developer Guide](https://kserve.github.io/website/docs/developer-guide)
*   **Contributor Guide:** [Contributor Guide](https://kserve.github.io/website/docs/developer-guide/contribution)
*   **Adopters:** [Adopters](https://kserve.github.io/website/docs/community/adopters)