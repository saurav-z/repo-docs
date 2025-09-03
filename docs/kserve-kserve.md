# KServe: Production-Ready Model Serving on Kubernetes

**KServe empowers you to easily deploy and manage your machine learning models in production on Kubernetes, providing advanced features and optimized performance.**  [See the original repository](https://github.com/kserve/kserve).

## Key Features

*   **Simplified Deployment:** KServe offers a Kubernetes Custom Resource Definition (CRD) to streamline the deployment of machine learning models, including TensorFlow, XGBoost, PyTorch, and Hugging Face models.
*   **Standardized Inference Protocols:** Supports standardized inference protocols for consistent performance across different ML frameworks, including the OpenAI specification for generative models.
*   **Serverless Inference:** Enables serverless inference workloads with request-based autoscaling, including scale-to-zero, on both CPU and GPU resources.
*   **Advanced Autoscaling and Resource Management:** Offers GPU autoscaling, scale-to-zero, and canary rollouts for efficient resource utilization.
*   **ModelMesh Integration:** Provides high scalability, density packing, and intelligent routing using ModelMesh.
*   **Comprehensive Serving Capabilities:**  Supports pre-processing, post-processing, monitoring, and explainability for a complete production ML serving solution.
*   **Advanced Deployment Strategies:**  Provides deployment options for canary rollouts, pipeline and ensembles using InferenceGraph.

## Why Choose KServe?

*   **Cloud-Agnostic Model Serving:** KServe is a standard, cloud-agnostic platform for serving predictive and generative AI models on Kubernetes.
*   **High Performance:**  Offers optimized inference protocols for efficient model serving.
*   **Scalable and Flexible:** Provides serverless capabilities and autoscaling for adapting to varying workloads.
*   **Production-Ready:** Delivers a complete solution for production ML, including pre/post-processing, monitoring, and explainability.

## Getting Started

*   **[Installation Guides](https://kserve.github.io/website/docs/admin-guide/overview)**: Choose from Serverless, Raw Kubernetes, ModelMesh and quick local installation options.
*   **[Kubeflow Integration](https://www.kubeflow.org/docs/external-add-ons/kserve/kserve)**: KServe is a key component of Kubeflow, with dedicated documentation.
*   **[Create your first InferenceService](https://kserve.github.io/website/docs/getting-started/genai-first-isvc)**

## Resources

*   **[KServe Website](https://kserve.github.io/website/)**: For detailed documentation, tutorials, and community information.
*   **[Roadmap](./ROADMAP.md)**: Learn about future developments and planned features.
*   **[InferenceService API Reference](https://kserve.github.io/website/docs/reference/crd-api)**: Explore the API documentation for InferenceService.
*   **[Developer Guide](https://kserve.github.io/website/docs/developer-guide)**: Understand how to contribute and extend KServe.
*   **[Contributor Guide](https://kserve.github.io/website/docs/developer-guide/contribution)**: Learn how to contribute to the project.
*   **[Adopters](https://kserve.github.io/website/docs/community/adopters)**: See which organizations are using KServe.