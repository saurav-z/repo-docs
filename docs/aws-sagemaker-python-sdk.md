<div align="center">
  <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">
  <h1>Amazon SageMaker Python SDK: Train and Deploy ML Models</h1>
  <p><i>Effortlessly build, train, and deploy machine learning models on Amazon SageMaker.</i></p>
  <a href="https://github.com/aws/sagemaker-python-sdk">
    <img src="https://img.shields.io/github/stars/aws/sagemaker-python-sdk?style=social" alt="GitHub stars">
  </a>
</div>

## Overview

The SageMaker Python SDK is your key to unlocking the power of Amazon SageMaker for machine learning. This open-source library simplifies the entire ML lifecycle, from model training and evaluation to deployment and monitoring.  Leverage popular deep learning frameworks, Amazon algorithms, and your own custom algorithms to build and deploy sophisticated models at scale.  Visit the [original repository](https://github.com/aws/sagemaker-python-sdk) for the latest updates and contributions.

## Key Features

*   **Framework Support:** Train models using Apache MXNet, TensorFlow, PyTorch, and more.
*   **Built-in Algorithms:** Utilize scalable Amazon algorithms optimized for SageMaker and GPU training.
*   **Custom Algorithms:** Easily integrate your own algorithms built into SageMaker-compatible Docker containers.
*   **Model Deployment:** Deploy models with just a few lines of code and monitor them in production.
*   **Model Tuning:** Utilize SageMaker's automatic model tuning capabilities to find the best performing models.
*   **Batch Transform:** Perform batch transformations for large datasets quickly and efficiently.
*   **Integration with popular ML frameworks** Scikit-learn, XGBoost, Chainer, and more

## Getting Started

### Installing the SageMaker Python SDK

Install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Operating Systems

*   Unix/Linux
*   Mac

### Supported Python Versions

The SDK is tested and supports the following Python versions:

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

## Additional Resources

*   **Documentation:** [Read the Docs](https://sagemaker.readthedocs.io/en/stable/) - Comprehensive documentation, including the API reference.
*   **SageMaker SparkML Serving:** Deploy SparkML models using the `SparkMLModel` class.
*   **Examples and Tutorials:** Explore the [Read the Docs](https://sagemaker.readthedocs.io/en/stable/) for example code and tutorials.

## Telemetry

The SDK includes telemetry to help us understand user needs, diagnose issues, and deliver new features.  You can opt-out of telemetry by configuring the `TelemetryOptOut` parameter in the SDK defaults. See [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for detailed instructions.

## AWS Permissions

As a managed service, Amazon SageMaker needs permissions to perform operations on your behalf.  Learn more about the required permissions in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License. See [http://aws.amazon.com/apache2.0/](http://aws.amazon.com/apache2.0/) for the full license.

## Testing and Contributing

The SageMaker Python SDK provides unit and integration tests.  See the original README in the [GitHub repository](https://github.com/aws/sagemaker-python-sdk) for instructions on running tests and contributing to the project.