<!-- Banner Image -->
<img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">

# Amazon SageMaker Python SDK: Simplify Machine Learning on AWS

**The Amazon SageMaker Python SDK provides a powerful and flexible way to build, train, and deploy machine learning models directly on Amazon SageMaker. [Visit the original repository](https://github.com/aws/sagemaker-python-sdk)**

<!-- Badges -->
[![PyPI Version](https://img.shields.io/pypi/v/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Conda-Forge Version](https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg)](https://anaconda.org/conda-forge/sagemaker-python-sdk)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)
[![Documentation Status](https://readthedocs.org/projects/sagemaker/badge/?version=stable)](https://sagemaker.readthedocs.io/en/stable/)
[![CI Health](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg)](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml)

## Key Features

*   **Simplified Model Training:** Train models using popular deep learning frameworks like Apache MXNet and TensorFlow, or with your own custom algorithms in Docker containers.
*   **Built-in Algorithms:** Utilize scalable, pre-built Amazon algorithms optimized for SageMaker and GPU training.
*   **Flexible Deployment:** Deploy trained models to SageMaker endpoints for real-time or batch predictions.
*   **Integration with SageMaker Services:** Leverage features like Automatic Model Tuning, Batch Transform, Model Monitoring, and Debugger for comprehensive model lifecycle management.
*   **SparkML Serving:** Easily deploy and perform predictions against SparkML models serialized with the MLeap library.

## Getting Started

### Installation

Install the latest version of the SDK using pip:

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

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

## SageMaker SparkML Serving

With SageMaker SparkML Serving, you can perform predictions against a SparkML Model in SageMaker. Models must be serialized using the MLeap library.

*Supported major version of Spark: 3.3 (MLeap version - 0.20.0)*

```python
from sagemaker.sparkml.model import SparkMLModel

sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more information, see `SageMaker SparkML Serving Container <https://github.com/aws/sagemaker-sparkml-serving-container>`.

## Documentation

For detailed information, including the API reference, visit the official documentation:  [Read the Docs](https://sagemaker.readthedocs.io/)

## AWS Permissions

Amazon SageMaker requires specific permissions to perform operations on your behalf. Review the  [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details on necessary permissions.

If you are using an IAM role with a path, you should grant permission for `iam:GetRole`.

## Telemetry

The `sagemaker` library has telemetry enabled to help us improve the SDK. If you want to opt out, set the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration: [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk)

## Testing

### Unit Tests

Run unit tests with:

```bash
tox tests/unit
```

### Integration Tests

Run integration tests with:

```bash
tox -- tests/integ
```

### Git Hooks

Enable git hooks by running:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.

```
http://aws.amazon.com/apache2.0/
```