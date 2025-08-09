<!-- PROJECT LOGO -->
<p align="center">
  <a href="https://github.com/aws/sagemaker-python-sdk">
    <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="Logo" height="100">
  </a>

  <h3 align="center">Amazon SageMaker Python SDK</h3>

  <p align="center">
    Supercharge your machine learning workflows with the official Amazon SageMaker Python SDK.
    <br />
    <a href="https://github.com/aws/sagemaker-python-sdk"><strong>Explore the docs »</strong></a>
  </p>
</p>

[![PyPI Latest Version](https://img.shields.io/pypi/v/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Conda-Forge Version](https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg)](https://anaconda.org/conda-forge/sagemaker-python-sdk)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)
[![Documentation Status](https://readthedocs.org/projects/sagemaker/badge/?version=stable)](https://sagemaker.readthedocs.io/en/stable/)
[![CI Health](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg)](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml)

## About the Amazon SageMaker Python SDK

The Amazon SageMaker Python SDK is an open-source library designed to simplify and streamline the process of building, training, and deploying machine learning models on Amazon SageMaker. This SDK provides a high-level, Pythonic interface to the SageMaker platform, allowing you to focus on your machine learning tasks rather than infrastructure management.

### Key Features

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Leverage scalable, optimized implementations of core machine learning algorithms.
*   **Bring Your Own Algorithms (BYOA):** Train and host models using your custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Model Deployment:** Easily deploy your trained models for real-time inference.
*   **Model Training:** Facilitates distributed training on SageMaker.
*   **Integration:** Seamlessly integrates with other AWS services.

## Getting Started

### Installing the SageMaker Python SDK

Install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker
```
or from source:
```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

For detailed installation and usage instructions, refer to the [official documentation](https://sagemaker.readthedocs.io/en/stable/).

### Supported Operating Systems

*   Unix/Linux
*   Mac

### Supported Python Versions

The SDK is tested on the following Python versions:

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

## Telemetry

The `sagemaker` library has telemetry enabled to help us better understand user needs, diagnose issues, and deliver new features. This telemetry tracks the usage of various SageMaker functions.

If you prefer to opt out of telemetry, you can easily do so by setting the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration. For detailed instructions, please visit [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk).

## AWS Permissions

Amazon SageMaker is a managed service, and it operates on your behalf using AWS hardware. SageMaker can perform only those operations allowed by your user permissions. You can find more details on permissions in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

The SageMaker Python SDK should not require additional permissions beyond what's needed to use SageMaker. However, if you use an IAM role with a path, grant permission for `iam:GetRole`.

## Licensing

The SageMaker Python SDK is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).

## Testing

### Unit Tests

Run unit tests with `tox`:

```bash
tox tests/unit
```

### Integration Tests

Before running the integration tests, make sure you have the prerequisites met:

1.  AWS account credentials must be available in the environment for the boto3 client to use.
2.  The AWS account should have an IAM role named `SageMakerRole`. It should have the AmazonSageMakerFullAccess policy attached and a policy that grants the necessary permissions to use Elastic Inference (see the  [Elastic Inference setup documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/ei-setup.html)).
3. To run remote_function tests, a dummy ECR repo should be created with the following command:
    ```bash
    aws ecr create-repository --repository-name remote-function-dummy-container
    ```

To run integration tests, you can selectively run individual tests or all tests:

```bash
# Run a specific test
tox -- -k 'test_i_care_about'

# Run all tests (may take a while)
tox -- tests/integ

# Run tests in parallel
tox -- -n auto tests/integ
```

## Git Hooks

To enable all git hooks, run these commands in the repository directory:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

## Building Sphinx Documentation

To build the Sphinx documentation:

```bash
# Install dependencies (using conda)
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

# Install dependencies (using pip)
pip install -r doc/requirements.txt

# Install local version
pip install --upgrade .

# Build the documentation
cd sagemaker-python-sdk/doc
make html

# View the documentation (in your browser)
cd _build/html
python -m http.server 8000
```

## SageMaker SparkML Serving

With SageMaker SparkML Serving, you can perform predictions against a SparkML Model in SageMaker. The model should be serialized with the ``MLeap`` library.

For details on MLeap, see [https://github.com/combust/mleap](https://github.com/combust/mleap).

Supported Spark major version: 3.3 (MLeap version - 0.20.0)

Example of creating an instance of `SparkMLModel` class and using the `deploy()` method:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

Invoke the endpoint with a `CSV` payload:

```python
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more information about `content-type`, `Accept` formats and the `schema`, see [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).