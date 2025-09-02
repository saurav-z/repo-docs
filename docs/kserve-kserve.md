# KServe: The Open-Source Platform for Production Machine Learning Serving

KServe empowers you to effortlessly deploy, manage, and scale your machine learning models on Kubernetes, simplifying the complexities of production AI. [Learn more about KServe on GitHub](https://github.com/kserve/kserve).

[![go.dev reference](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/kserve/kserve)
[![Coverage Status](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/andyi2it/5174bd748ac63a6e4803afea902e9810/raw/coverage.json)](https://github.com/kserve/kserve/actions/workflows/go.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/kserve/kserve)](https://goreportcard.com/report/github.com/kserve/kserve)
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/6643/badge)](https://bestpractices.coreinfrastructure.org/projects/6643)
[![Releases](https://img.shields.io/github/release-pre/kserve/kserve.svg?sort=semver)](https://github.com/kserve/kserve/releases)
[![LICENSE](https://img.shields.io/github/license/kserve/kserve.svg)](https://github.com/kserve/kserve/blob/master/LICENSE)
[![Slack Status](https://img.shields.io/badge/slack-join_chat-white.svg?logo=slack&style=social)](https://github.com/kserve/community/blob/main/README.md#questions-and-issues)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20KServe%20Guru-006BFF)](https://gurubase.io/g/kserve)

## Key Features

*   **Standardized Inference Protocol:** Offers a performant and standardized inference protocol across various ML frameworks, including the OpenAI specification for generative models.
*   **Serverless Inference Workloads:** Supports modern serverless inference with request-based autoscaling, including scale-to-zero for CPU and GPU resources.
*   **High Scalability and Density:** Provides high scalability, density packing, and intelligent routing through ModelMesh integration.
*   **Simplified Production Serving:** Enables simple, pluggable production serving for inference, pre/post-processing, monitoring, and explainability.
*   **Advanced Deployment Strategies:** Supports advanced deployment patterns like canary rollouts, pipelines, and ensembles using InferenceGraph.
*   **Kubernetes Native:** Leverages Kubernetes Custom Resource Definitions (CRDs) for seamless integration with your existing infrastructure.
*   **Broad Framework Support:**  Works with popular frameworks like TensorFlow, XGBoost, Scikit-learn, PyTorch, and Hugging Face Transformers/LLMs.

## Why Choose KServe?

KServe is designed to streamline your machine learning model serving pipeline. It abstracts away the complexities of infrastructure management, allowing you to focus on model development and deployment.  Key benefits include:

*   **Cloud Agnostic:** Deploy your models on any Kubernetes cluster.
*   **Optimized Performance:** Built for high-performance, scalable model serving.
*   **Simplified Management:** Reduces operational overhead with automated scaling, health checks, and more.
*   **Extensible and Adaptable:** Easily integrates with your existing ML workflow and tools.

## Installation

KServe offers flexible installation options to suit your needs:

*   **[Serverless Installation](https://kserve.github.io/website/docs/admin-guide/overview#serverless-deployment):**  Leverages Knative for serverless deployments.
*   **[Raw Kubernetes Deployment](https://kserve.github.io/website/docs/admin-guide/overview#raw-kubernetes-deployment):** A lightweight alternative.
*   **[ModelMesh Installation](https://kserve.github.io/website/docs/admin-guide/overview#modelmesh-deployment):**  For high-scale, high-density model serving scenarios.
*   **[Quick Installation](https://kserve.github.io/website/docs/getting-started/quickstart-guide):** Get started quickly with a local installation.
*   **Kubeflow Installation:** KServe is a key component of Kubeflow.  See the [Kubeflow KServe documentation](https://www.kubeflow.org/docs/external-add-ons/kserve/kserve) for details.  Specific guides are available for running [on AWS](https://awslabs.github.io/kubeflow-manifests/main/docs/component-guides/kserve) and [on OpenShift Container Platform](https://github.com/kserve/kserve/blob/master/docs/OPENSHIFT_GUIDE.md).

## Get Started

*   **[Create your first InferenceService](https://kserve.github.io/website/docs/getting-started/genai-first-isvc)**
*   **[KServe Website Documentation](https://kserve.github.io/website)**
*   **[Presentations and Demos](https://kserve.github.io/website/docs/community/presentations)**

## Resources

*   [Roadmap](./ROADMAP.md)
*   [InferenceService API Reference](https://kserve.github.io/website/docs/reference/crd-api)
*   [Developer Guide](https://kserve.github.io/website/docs/developer-guide)
*   [Contributor Guide](https://kserve.github.io/website/docs/developer-guide/contribution)
*   [Adopters](https://kserve.github.io/website/docs/community/adopters)