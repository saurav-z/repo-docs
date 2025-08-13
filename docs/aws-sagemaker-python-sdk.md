<!-- Banner Image -->
<img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">

# SageMaker Python SDK: Train, Deploy, and Manage Machine Learning Models with Ease

**The SageMaker Python SDK provides a powerful and flexible toolkit for building, training, and deploying machine learning models on Amazon SageMaker.**  [Explore the original repository](https://github.com/aws/sagemaker-python-sdk).

<!-- Badges -->
[![PyPI Version](https://img.shields.io/pypi/v/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Conda-Forge Version](https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg)](https://anaconda.org/conda-forge/sagemaker-python-sdk)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)
[![Documentation Status](https://readthedocs.org/projects/sagemaker/badge/?version=stable)](https://sagemaker.readthedocs.io/en/stable/)
[![CI Health](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg)](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml)

## Key Features

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Leverage scalable, optimized Amazon algorithms for core machine learning tasks.
*   **Custom Algorithms:**  Bring your own algorithms built within SageMaker compatible Docker containers.
*   **Model Deployment:** Easily deploy your trained models to SageMaker endpoints for real-time predictions.
*   **Integration with MLeap for SparkML:**  Deploy SparkML models serialized with the MLeap library.
*   **Comprehensive Documentation:** Detailed documentation available at [Read the Docs](https://sagemaker.readthedocs.io).

## Table of Contents

1.  [Installing the SageMaker Python SDK](#installing-the-sagemaker-python-sdk)
2.  [Using the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html)
3.  [Using MXNet](https://sagemaker.readthedocs.io/en/stable/using_mxnet.html)
4.  [Using TensorFlow](https://sagemaker.readthedocs.io/en/stable/using_tf.html)
5.  [Using Chainer](https://sagemaker.readthedocs.io/en/stable/using_chainer.html)
6.  [Using PyTorch](https://sagemaker.readthedocs.io/en/stable/using_pytorch.html)
7.  [Using Scikit-learn](https://sagemaker.readthedocs.io/en/stable/using_sklearn.html)
8.  [Using XGBoost](https://sagemaker.readthedocs.io/en/stable/using_xgboost.html)
9.  [SageMaker Reinforcement Learning Estimators](https://sagemaker.readthedocs.io/en/stable/using_rl.html)
10. [SageMaker SparkML Serving](#sagemaker-sparkml-serving)
11. [Amazon SageMaker Built-in Algorithm Estimators](src/sagemaker/amazon/README.rst)
12. [Using SageMaker AlgorithmEstimators](https://sagemaker.readthedocs.io/en/stable/overview.html#using-sagemaker-algorithmestimators)
13. [Consuming SageMaker Model Packages](https://sagemaker.readthedocs.io/en/stable/overview.html#consuming-sagemaker-model-packages)
14. [BYO Docker Containers with SageMaker Estimators](https://sagemaker.readthedocs.io/en/stable/overview.html#byo-docker-containers-with-sagemaker-estimators)
15. [SageMaker Automatic Model Tuning](https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-automatic-model-tuning)
16. [SageMaker Batch Transform](https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-batch-transform)
17. [Secure Training and Inference with VPC](https://sagemaker.readthedocs.io/en/stable/overview.html#secure-training-and-inference-with-vpc)
18. [BYO Model](https://sagemaker.readthedocs.io/en/stable/overview.html#byo-model)
19. [Inference Pipelines](https://sagemaker.readthedocs.io/en/stable/overview.html#inference-pipelines)
20. [Amazon SageMaker Operators in Apache Airflow](https://sagemaker.readthedocs.io/en/stable/using_workflow.html)
21. [SageMaker Autopilot](src/sagemaker/automl/README.rst)
22. [Model Monitoring](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_model_monitoring.html)
23. [SageMaker Debugger](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html)
24. [SageMaker Processing](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html)

## Installing the SageMaker Python SDK

Install the latest version from PyPI using pip:

```bash
pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

## Supported Operating Systems

*   Unix/Linux
*   Mac

## Supported Python Versions

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

## Telemetry

The `sagemaker` library includes telemetry to help improve functionality and address issues.  You can opt out by setting the `TelemetryOptOut` parameter to `true` in your SDK configuration.  See [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for details.

## AWS Permissions

Amazon SageMaker requires specific AWS permissions. Refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for detailed information on necessary permissions. Ensure your IAM role has the required permissions for SageMaker and, if applicable, permission for `iam:GetRole` if your role has a path.

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License. Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. The license is available at:
http://aws.amazon.com/apache2.0/

## Running Tests

### Unit Tests

Run unit tests using tox:

```bash
tox tests/unit
```

### Integration Tests

Prerequisites:
1. AWS account credentials.
2. IAM role named `SageMakerRole` with `AmazonSageMakerFullAccess` and Elastic Inference permissions.
3.  Dummy ECR repo created :  `aws ecr create-repository --repository-name remote-function-dummy-container`

Run individual integration tests:

```bash
tox -- -k 'test_i_care_about'
```

Run all integration tests sequentially:

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

Enable an individual hook by moving it from `.githooks/` to `.git/hooks/`.

## Building Sphinx Docs

1.  Set up a Python environment and install dependencies (see `doc/requirements.txt`):

    ```bash
    # conda
    conda create -n sagemaker python=3.12
    conda activate sagemaker
    conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

    # pip
    pip install -r doc/requirements.txt
    ```

2.  Install the SDK locally:

    ```bash
    pip install --upgrade .
    ```

3.  Build the documentation:

    ```bash
    cd sagemaker-python-sdk/doc
    make html
    ```

4.  Preview the website:

    ```bash
    cd _build/html
    python -m http.server 8000
    ```

    Visit http://localhost:8000 in your browser.

## SageMaker SparkML Serving

Deploy SparkML models using MLeap for real-time predictions.  SparkML models must be serialized with the `MLeap` library.

Supported Spark version: 3.3 (MLeap version - 0.20.0)

Example:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

See [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container) for details on supported content types, accept formats, and schema structures.