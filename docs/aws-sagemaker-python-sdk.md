<!-- Banner Image -->
<img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">

# SageMaker Python SDK: Train and Deploy Machine Learning Models with Ease

**The SageMaker Python SDK empowers developers to build, train, and deploy machine learning models efficiently on Amazon SageMaker.**  Explore the power of Amazon SageMaker with this open-source Python library. ([See the original repo](https://github.com/aws/sagemaker-python-sdk))

## Key Features

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Leverage scalable, optimized implementations of core machine learning algorithms.
*   **Bring Your Own Algorithms (BYOA):** Train and host models with custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Model Deployment:** Easily deploy your trained models for real-time or batch inference.
*   **Integration with MLeap:** Deploy SparkML models serialized with the MLeap library using the SageMaker SparkML Serving feature.
*   **Comprehensive Documentation:** Access detailed documentation, including API references.
*   **Automatic Model Tuning:** Optimize your models with SageMaker's automatic model tuning capabilities.
*   **Batch Transform:** Perform batch transformations on your data using trained models.
*   **Model Monitoring:** Monitor your models for data drift and other performance issues.
*   **Secure Training and Inference:** Secure your training and inference workloads with VPC support.

## Core Functionality and Usage

The SageMaker Python SDK allows you to streamline your machine learning workflow from start to finish. Here's how:

*   **Training:**  The SDK provides tools to train models on SageMaker using various algorithms and frameworks.
*   **Deployment:** Easily deploy your trained models to SageMaker endpoints for real-time or batch inference.
*   **Model Management:** Manage your models, endpoints, and training jobs through a unified interface.

## Getting Started

### Installation

Install the latest version of the SageMaker Python SDK using `pip`:

```bash
pip install sagemaker
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Python Versions

The SageMaker Python SDK is tested on:

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

## Resources

*   **Documentation:** [Read the Docs](https://sagemaker.readthedocs.io/)
*   **GitHub Repository:** [aws/sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk)

## Telemetry

The SDK includes telemetry to help improve user experience. You can opt-out by setting the `TelemetryOptOut` parameter to `true`. More details on configuring defaults are available in the documentation.

## AWS Permissions

Amazon SageMaker operates on your behalf within your AWS account.  For successful operation, your IAM role requires appropriate permissions. See the AWS Documentation for details.

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.