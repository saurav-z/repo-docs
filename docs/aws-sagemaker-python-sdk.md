<img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100px">

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models

**Easily train and deploy your machine learning models on Amazon SageMaker with the open-source SageMaker Python SDK!**  [Get started with the SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk).

## Key Features

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Leverage scalable, optimized implementations of core machine learning algorithms.
*   **Custom Algorithms:** Bring your own algorithms built into SageMaker compatible Docker containers.
*   **Model Deployment:** Simplify the process of deploying your trained models for real-time or batch predictions.
*   **Integration with MLeap:** Seamlessly deploy and perform predictions with SparkML models serialized with MLeap.
*   **Comprehensive Documentation:** Access detailed documentation, including API references, on [Read the Docs](https://sagemaker.readthedocs.io/).

## What is the SageMaker Python SDK?

The SageMaker Python SDK is a powerful open-source library designed to streamline the machine learning workflow on Amazon SageMaker.  It empowers data scientists and developers to efficiently train, deploy, and manage machine learning models.  This SDK provides a user-friendly interface to interact with SageMaker's features, allowing for faster experimentation and production deployment.

### Core Capabilities:

*   **Model Training:** Facilitates model training using a variety of algorithms and frameworks.
*   **Model Deployment:** Simplifies the deployment of trained models to production.
*   **Scalability:** Enables scaling of model training and deployment resources as needed.
*   **Monitoring & Management:** Provides tools for monitoring model performance and managing model versions.

## Getting Started

### Installation

Install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker
```

Alternatively, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Python Versions

The SDK is tested on:

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

## Additional Resources

*   **Documentation:**  [Read the Docs](https://sagemaker.readthedocs.io/) - Comprehensive documentation with API reference.
*   **SageMaker SparkML Serving Container:** [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container) - More information about the different ``content-type`` and ``Accept`` formats as well as the structure of the ``schema`` that SageMaker SparkML Serving recognizes
*   **Licensing:** [Apache 2.0 License](http://aws.amazon.com/apache2.0/)
*   **Original Repository:**  [https://github.com/aws/sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk)