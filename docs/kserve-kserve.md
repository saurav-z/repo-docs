# KServe: Production Machine Learning Model Serving on Kubernetes

**KServe is a Kubernetes-native platform that simplifies and accelerates the deployment, management, and scaling of machine learning models for production environments.** ([Original Repository](https://github.com/kserve/kserve))

KServe provides a standardized interface for serving predictive and generative AI models, enabling seamless integration with various machine learning frameworks and addressing key production serving challenges.

## Key Features:

*   **Kubernetes-Native:** Leverages Kubernetes Custom Resource Definitions (CRDs) for model serving.
*   **Framework Agnostic:** Supports TensorFlow, XGBoost, Scikit-learn, PyTorch, Hugging Face, and other models.
*   **Standardized Inference Protocol:** Implements performant and standardized inference protocols, including the OpenAI specification for generative models.
*   **Serverless Inference:** Supports serverless workloads with request-based autoscaling, including scale-to-zero on both CPU and GPU.
*   **Advanced Deployment Strategies:** Enables canary rollouts, pipeline integrations, and ensemble deployments through InferenceGraph.
*   **Scalability and Density:** Provides high scalability and density packing using ModelMesh.
*   **Production-Ready Features:** Offers health checks, autoscaling, networking, and server configuration, simplifying deployment and management.
*   **Extensibility:** Supports pre-processing, post-processing, monitoring, and explainability.

## Why Choose KServe?

KServe addresses the core challenges of production machine learning, offering:

*   **Simplified Deployment:** Reduces the complexity of deploying and managing ML models on Kubernetes.
*   **Scalability and Efficiency:** Optimizes resource utilization through autoscaling and density packing.
*   **Standardized Interface:** Provides a consistent interface across different ML frameworks and deployment environments.
*   **Production-Grade Features:** Includes features like health checks, canary deployments, and model explainability.

## Installation

KServe can be installed in several ways:

*   **Serverless Installation:** Leverages Knative for serverless deployment.
*   **Raw Kubernetes Deployment:** A lightweight installation option.
*   **ModelMesh Installation:** Enables high-scale and high-density model serving.
*   **Quick Installation:** Get up and running locally with a quickstart guide.
*   **Kubeflow Integration:** Seamless integration with Kubeflow.

For detailed installation instructions, refer to the [KServe website documentation](https://kserve.github.io/website/docs/admin-guide/overview).

## Get Started

*   **Create your first InferenceService:** [Getting Started Guide](https://kserve.github.io/website/docs/getting-started/genai-first-isvc)
*   **Roadmap:** [ROADMAP.md](./ROADMAP.md)
*   **InferenceService API Reference:** [KServe Website API Docs](https://kserve.github.io/website/docs/reference/crd-api)
*   **Developer Guide:** [KServe Website Developer Guide](https://kserve.github.io/website/docs/developer-guide)
*   **Contributor Guide:** [KServe Website Contributor Guide](https://kserve.github.io/website/docs/developer-guide/contribution)
*   **Adopters:** [KServe Website Adopters](https://kserve.github.io/website/docs/community/adopters)