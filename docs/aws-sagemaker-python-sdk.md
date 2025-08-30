<div align="center">
  <a href="https://github.com/aws/sagemaker-python-sdk">
    <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" width="300">
  </a>
  <h1>SageMaker Python SDK: Train, Deploy, and Manage Machine Learning Models</h1>
  <p><em>Simplify your machine learning workflow with the official SageMaker Python SDK.</em></p>
  <p>
    <a href="https://pypi.org/project/sagemaker">
      <img src="https://img.shields.io/pypi/v/sagemaker.svg" alt="Latest Version">
    </a>
    <a href="https://anaconda.org/conda-forge/sagemaker-python-sdk">
      <img src="https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg" alt="Conda-Forge Version">
    </a>
    <a href="https://pypi.python.org/pypi/sagemaker">
      <img src="https://img.shields.io/pypi/pyversions/sagemaker.svg" alt="Supported Python Versions">
    </a>
    <a href="https://github.com/python/black">
      <img src="https://img.shields.io/badge/code_style-black-000000.svg" alt="Code style: black">
    </a>
    <a href="https://sagemaker.readthedocs.io/en/stable/">
      <img src="https://readthedocs.org/projects/sagemaker/badge/?version=stable" alt="Documentation Status">
    </a>
    <a href="https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml">
      <img src="https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg" alt="CI Health">
    </a>
  </p>
</div>

## Overview

The SageMaker Python SDK is an open-source library that simplifies building, training, and deploying machine learning models on Amazon SageMaker. This SDK provides a user-friendly interface to leverage the power of SageMaker for your machine learning projects, including supporting the training and deployment of models using popular frameworks and algorithms.

**Key Features:**

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Built-in Algorithms:** Utilize Amazon's scalable, pre-built machine learning algorithms optimized for SageMaker and GPU training.
*   **Custom Algorithm Support:** Easily train and host models with your own algorithms packaged in SageMaker-compatible Docker containers.
*   **SparkML Integration:** Deploy and predict with SparkML models using MLeap for seamless integration.
*   **Comprehensive Functionality:** Includes features for automatic model tuning, batch transform, model monitoring, and more.
*   **Simplified Workflow:** Streamline your machine learning lifecycle from development to deployment with Pythonic interfaces.

## Getting Started

### Installing the SageMaker Python SDK

Install the latest version of the SageMaker Python SDK using `pip`:

```bash
pip install sagemaker==<Latest version from https://pypi.org/project/sagemaker/>
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Platforms and Versions

*   **Supported Operating Systems:** Unix/Linux and Mac.
*   **Supported Python Versions:**
    *   Python 3.9
    *   Python 3.10
    *   Python 3.11
    *   Python 3.12

## AWS Permissions

Amazon SageMaker requires specific permissions to perform operations on your behalf within your AWS account.  Consult the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for detailed information.

## Telemetry

The `sagemaker` library includes telemetry to help improve the SDK. To opt-out, set the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration.  See the [SageMaker Python SDK documentation](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for instructions.

## SparkML Serving

You can use the SageMaker Python SDK to perform predictions against a SparkML Model hosted in SageMaker. Serialize your SparkML model with the MLeap library.

Example:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

## Testing

To run the tests, install testing dependencies:

```bash
pip install --upgrade .[test]
```

or, if you're using Zsh:
```bash
pip install --upgrade .\[test\]
```

### Unit Tests

Run unit tests with `tox`:

```bash
tox tests/unit
```

### Integration Tests

To run integration tests, configure AWS credentials and an IAM role named `SageMakerRole`.

Run all integration tests:

```bash
tox -- tests/integ
```

Or, run integration tests in parallel:

```bash
tox -- -n auto tests/integ
```

## Building Documentation

Build the Sphinx documentation:

1.  Install dependencies, see the `doc/requirements.txt` file for details.
2.  Navigate to the `doc` directory and run `make html`.
3.  View the docs in `doc/_build/html`.

## License

This SDK is licensed under the Apache 2.0 License.  See the full license at [http://aws.amazon.com/apache2.0/](http://aws.amazon.com/apache2.0/)

**[View the SageMaker Python SDK Repository on GitHub](https://github.com/aws/sagemaker-python-sdk)**