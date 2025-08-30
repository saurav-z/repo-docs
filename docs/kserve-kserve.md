# KServe: Production-Ready Machine Learning Serving on Kubernetes

**KServe simplifies the deployment and management of machine learning models on Kubernetes, providing a scalable and efficient solution for your AI needs.**  ([Original Repo](https://github.com/kserve/kserve))

KServe is a powerful open-source platform designed for serving machine learning (ML) models on Kubernetes.  It leverages Kubernetes Custom Resource Definitions (CRDs) to streamline the deployment and management of predictive and generative AI models, offering a robust solution for production environments.

## Key Features

*   **Standardized Inference Protocol:** Supports a standardized inference protocol across various ML frameworks, including the OpenAI specification for generative models.
*   **Serverless Inference:** Enables serverless inference workloads with request-based autoscaling, including scale-to-zero on both CPU and GPU resources.
*   **High Scalability and Density:** Offers high scalability and density packing options using ModelMesh for optimized resource utilization.
*   **Simplified Production Serving:** Provides a simple and pluggable solution for production ML serving, encompassing inference, pre/post-processing, monitoring, and explainability.
*   **Advanced Deployment Options:** Supports advanced deployment strategies like canary rollouts, pipelines, and ensembles with InferenceGraph.
*   **Framework Support:**  Works seamlessly with popular frameworks such as TensorFlow, XGBoost, Scikit-Learn, PyTorch, and Hugging Face Transformers/LLMs.

## Why Choose KServe?

KServe provides a cloud-agnostic solution for serving both predictive and generative AI models, built for highly scalable use cases. It encapsulates the complexities of autoscaling, networking, health checks, and server configuration, allowing you to focus on your models. With KServe, you can achieve:

*   **Efficient Resource Utilization:** Optimizing resource utilization with features like GPU autoscaling.
*   **Simplified Management:**  Streamlining the deployment and management of your ML models.
*   **Faster Time to Production:** Accelerating the process of bringing your models to production.

## Installation

KServe offers flexible installation options to suit your needs:

*   **Serverless Installation:** Leverages Knative for serverless deployment of InferenceService.
*   **Raw Kubernetes Deployment:** A lightweight alternative to serverless, ideal for specific use cases.
*   **ModelMesh Installation:** Enables high-scale, high-density, and frequently changing model serving.
*   **Quick Installation:**  Install KServe on your local machine for testing and development.

KServe is also a key component of Kubeflow. Refer to the [Kubeflow KServe documentation](https://www.kubeflow.org/docs/external-add-ons/kserve/kserve) for more information and guides on deploying KServe on AWS or OpenShift Container Platform.

## Get Started

*   **[Create your first InferenceService](https://kserve.github.io/website/docs/getting-started/genai-first-isvc)**
*   **[KServe website documentation](https://kserve.github.io/website)** for detailed guides and features.

## Additional Resources

*   **[Roadmap](./ROADMAP.md)**
*   **[InferenceService API Reference](https://kserve.github.io/website/docs/reference/crd-api)**
*   **[Developer Guide](https://kserve.github.io/website/docs/developer-guide)**
*   **[Contributor Guide](https://kserve.github.io/website/docs/developer-guide/contribution)**
*   **[Adopters](https://kserve.github.io/website/docs/community/adopters)**