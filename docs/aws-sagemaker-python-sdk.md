<!--  SageMaker Python SDK - Optimized README -->

[![SageMaker](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)](https://github.com/aws/sagemaker-python-sdk)

# SageMaker Python SDK: Simplify Machine Learning on AWS

**The SageMaker Python SDK empowers you to build, train, and deploy machine learning models at scale using Amazon SageMaker.**

[View the Original Repository on GitHub](https://github.com/aws/sagemaker-python-sdk)

<!-- Badges Section -->

[![PyPI Version](https://img.shields.io/pypi/v/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Conda-Forge Version](https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg)](https://anaconda.org/conda-forge/sagemaker-python-sdk)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)
[![Documentation Status](https://readthedocs.org/projects/sagemaker/badge/?version=stable)](https://sagemaker.readthedocs.io/en/stable/)
[![CI Health](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg)](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml)

## Key Features

*   **Simplified Model Training:** Easily train models with popular deep learning frameworks like **Apache MXNet** and **TensorFlow**.
*   **Built-in Algorithm Support:** Utilize scalable, optimized **Amazon algorithms** for core machine learning tasks.
*   **Bring Your Own Algorithms:** Train and deploy models using custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Framework Flexibility:** Support for a wide range of frameworks, including PyTorch, Scikit-learn, XGBoost, and Chainer.
*   **SparkML Integration:** Deploy and perform predictions against SparkML models using MLeap.
*   **Comprehensive Functionality:**  Includes support for Automatic Model Tuning, Batch Transform, Model Monitoring, Debugger, and more.
*   **Integration with Apache Airflow:** Leverage SageMaker operators within your Airflow workflows.

## Getting Started

### Installing the SageMaker Python SDK

Install the latest version using pip:

```bash
pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>
```

Or install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Operating Systems

The SDK supports Unix/Linux and Mac.

### Supported Python Versions

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

## Telemetry

The library has telemetry enabled to help us better understand user needs, diagnose issues, and deliver new features. This telemetry tracks the usage of various SageMaker functions.

If you prefer to opt out of telemetry, you can easily do so by setting the ``TelemetryOptOut`` parameter to ``true`` in the SDK defaults configuration. For detailed instructions, please visit `Configuring and using defaults with the SageMaker Python SDK <https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk>`__.

## AWS Permissions

Amazon SageMaker performs operations on your behalf on the AWS hardware that is managed by Amazon SageMaker.
Amazon SageMaker can perform only operations that the user permits.
You can read more about which permissions are necessary in the `AWS Documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`__.

The SageMaker Python SDK should not require any additional permissions aside from what is required for using SageMaker.
However, if you are using an IAM role with a path in it, you should grant permission for ``iam:GetRole``.

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License. It is copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. The license is available at:
http://aws.amazon.com/apache2.0/

## Running Tests

### Unit Tests

Run unit tests using tox:

```bash
tox tests/unit
```

### Integration Tests

**Prerequisites:**

1.  AWS account credentials are available in the environment.
2.  IAM role named `SageMakerRole` in the AWS account.
3.  Dummy ECR repo created (remote_function tests):
    ```bash
    aws ecr create-repository --repository-name remote-function-dummy-container
    ```

**Run specific tests:**

```bash
tox -- -k 'test_i_care_about'
```

**Run all integration tests:**

```bash
tox -- tests/integ
```

**Run integration tests in parallel:**

```bash
tox -- -n auto tests/integ
```

## Git Hooks

Enable all git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

Enable an individual hook: move from `.githooks/` to `.git/hooks/`.

## Building Sphinx Docs

1.  **Setup Python Environment:**
    ```bash
    # conda
    conda create -n sagemaker python=3.12
    conda activate sagemaker
    conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

    # pip
    pip install -r doc/requirements.txt
    ```
2.  Install the SDK (if you haven't already):
    ```bash
    pip install --upgrade .
    ```
3.  Build the Docs:
    ```bash
    cd sagemaker-python-sdk/doc
    make html
    ```
4.  Preview:
    ```bash
    cd _build/html
    python -m http.server 8000
    ```
    Visit [http://localhost:8000](http://localhost:8000)

## SageMaker SparkML Serving

Deploy and perform predictions against SparkML models serialized with MLeap.

**Supported Spark Version:** 3.3 (MLeap 0.20.0)

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more details, see:  [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container)

```