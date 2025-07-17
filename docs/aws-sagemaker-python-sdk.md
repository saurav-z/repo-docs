<!-- Banner Image - SEO Optimized -->
<p align="center">
  <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="Amazon SageMaker" height="100">
</p>

<!-- Badges for quick status -->
<p align="center">
  <a href="https://pypi.org/project/sagemaker">
    <img src="https://img.shields.io/pypi/v/sagemaker.svg" alt="PyPI Latest Version">
  </a>
  <a href="https://anaconda.org/conda-forge/sagemaker-python-sdk">
    <img src="https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg" alt="Conda-Forge Version">
  </a>
  <a href="https://pypi.org/project/sagemaker">
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

# SageMaker Python SDK: Train and Deploy Machine Learning Models

**The SageMaker Python SDK simplifies building, training, and deploying machine learning models on Amazon SageMaker.**  ([View on GitHub](https://github.com/aws/sagemaker-python-sdk))

## Key Features

*   **Framework Support:** Train and deploy models using popular deep learning frameworks, including:
    *   Apache MXNet
    *   TensorFlow
    *   PyTorch
    *   Chainer
    *   Scikit-learn
    *   XGBoost

*   **Amazon Algorithms:** Utilize scalable, optimized implementations of core machine learning algorithms.
*   **Custom Algorithms:** Bring your own algorithms built into SageMaker compatible Docker containers.
*   **SparkML Serving**: Perform predictions against a SparkML Model in SageMaker.

## Table of Contents

*   [Installing the SageMaker Python SDK](#installing-the-sagemaker-python-sdk)
*   [Using the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html)
*   [Using MXNet](https://sagemaker.readthedocs.io/en/stable/using_mxnet.html)
*   [Using TensorFlow](https://sagemaker.readthedocs.io/en/stable/using_tf.html)
*   [Using Chainer](https://sagemaker.readthedocs.io/en/stable/using_chainer.html)
*   [Using PyTorch](https://sagemaker.readthedocs.io/en/stable/using_pytorch.html)
*   [Using Scikit-learn](https://sagemaker.readthedocs.io/en/stable/using_sklearn.html)
*   [Using XGBoost](https://sagemaker.readthedocs.io/en/stable/using_xgboost.html)
*   [SageMaker Reinforcement Learning Estimators](https://sagemaker.readthedocs.io/en/stable/using_rl.html)
*   [SageMaker SparkML Serving](#sagemaker-sparkml-serving)
*   [Amazon SageMaker Built-in Algorithm Estimators](src/sagemaker/amazon/README.rst)
*   [Using SageMaker AlgorithmEstimators](https://sagemaker.readthedocs.io/en/stable/overview.html#using-sagemaker-algorithmestimators)
*   [Consuming SageMaker Model Packages](https://sagemaker.readthedocs.io/en/stable/overview.html#consuming-sagemaker-model-packages)
*   [BYO Docker Containers with SageMaker Estimators](https://sagemaker.readthedocs.io/en/stable/overview.html#byo-docker-containers-with-sagemaker-estimators)
*   [SageMaker Automatic Model Tuning](https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-automatic-model-tuning)
*   [SageMaker Batch Transform](https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-batch-transform)
*   [Secure Training and Inference with VPC](https://sagemaker.readthedocs.io/en/stable/overview.html#secure-training-and-inference-with-vpc)
*   [BYO Model](https://sagemaker.readthedocs.io/en/stable/overview.html#byo-model)
*   [Inference Pipelines](https://sagemaker.readthedocs.io/en/stable/overview.html#inference-pipelines)
*   [Amazon SageMaker Operators in Apache Airflow](https://sagemaker.readthedocs.io/en/stable/using_workflow.html)
*   [SageMaker Autopilot](src/sagemaker/automl/README.rst)
*   [Model Monitoring](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_model_monitoring.html)
*   [SageMaker Debugger](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html)
*   [SageMaker Processing](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html)

## Installing the SageMaker Python SDK

You can install the latest version using pip:

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

## Telemetry

The `sagemaker` library has telemetry enabled to help us understand user needs, diagnose issues, and deliver new features. This telemetry tracks the usage of various SageMaker functions.

If you prefer to opt out of telemetry, you can easily do so by setting the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration. For detailed instructions, please visit [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk).

## AWS Permissions

Amazon SageMaker performs operations on your behalf. You can read more about which permissions are necessary in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License. Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  The license is available at:  http://aws.amazon.com/apache2.0/

## Running Tests

The SageMaker Python SDK includes unit and integration tests.

Install test dependencies:

```bash
pip install --upgrade .[test]  # or  pip install --upgrade .\[test\] for Zsh
```

### Unit Tests

Run unit tests using tox:

```bash
tox tests/unit
```

### Integration Tests

Prerequisites for Integration Tests:

1.  AWS account credentials configured.
2.  IAM role named `SageMakerRole` with `AmazonSageMakerFullAccess` and permissions for Elastic Inference.
3.  Dummy ECR repository named `remote-function-dummy-container`.

Selectively run specific integration tests:

```bash
tox -- -k 'test_i_care_about'
```

Run all integration tests:

```bash
tox -- tests/integ
```

Run integration tests in parallel:

```bash
tox -- -n auto tests/integ
```

## Git Hooks

Enable git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

## Building Sphinx Docs

Prerequisites:

```bash
# conda
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

# pip
pip install -r doc/requirements.txt
pip install --upgrade .
```

Build the docs:

```bash
cd sagemaker-python-sdk/doc
make html
```

Preview:

```bash
cd _build/html
python -m http.server 8000
```

## SageMaker SparkML Serving

With SageMaker SparkML Serving, perform predictions against a SparkML Model in SageMaker, serialized with `MLeap`.

Supported Spark version: 3.3 (MLeap version - 0.20.0)

Example:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

Invoke endpoint with a CSV payload:

```python
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more details, see the [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).