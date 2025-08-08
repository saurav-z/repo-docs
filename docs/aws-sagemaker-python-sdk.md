<!-- Banner Image -->
<img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">

# SageMaker Python SDK: Simplify Machine Learning on AWS

**The SageMaker Python SDK simplifies building, training, and deploying machine learning models on Amazon SageMaker.** Explore the original repository [here](https://github.com/aws/sagemaker-python-sdk).

<!-- Badges -->
[![PyPI Version](https://img.shields.io/pypi/v/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Conda-Forge Version](https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg)](https://anaconda.org/conda-forge/sagemaker-python-sdk)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)
[![Documentation Status](https://readthedocs.org/projects/sagemaker/badge/?version=stable)](https://sagemaker.readthedocs.io/en/stable/)
[![CI Health](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg)](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml)

## Key Features

*   **Simplified Model Training:** Train models using popular deep learning frameworks like Apache MXNet and TensorFlow, as well as Amazon algorithms and custom Docker containers.
*   **Easy Model Deployment:** Deploy trained models to SageMaker for real-time or batch inference.
*   **Framework Support:** Extensive support for frameworks including TensorFlow, PyTorch, and more.
*   **Built-in Algorithms:** Leverage scalable implementations of core machine learning algorithms optimized for SageMaker.
*   **SparkML Serving:** Deploy SparkML models for predictions with MLeap serialization.
*   **Comprehensive Functionality:** Includes support for automatic model tuning, batch transform, model monitoring and debugging, and more.

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

Install the latest version using pip:

```bash
pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>
```

Alternatively, install from source:

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

### Telemetry

The `sagemaker` library has telemetry enabled to help us better understand user needs and deliver new features.  You can opt out by setting the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration. For more info, see [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk).

### AWS Permissions

Amazon SageMaker performs operations on your behalf, and the necessary permissions are documented in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html). The SageMaker Python SDK typically requires no additional permissions beyond what is required for using SageMaker, but if you use an IAM role with a path, grant permission for `iam:GetRole`.

### Licensing

SageMaker Python SDK is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/). It is copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

## Running Tests

### Unit Tests

Run unit tests using tox:

```bash
tox tests/unit
```

### Integration Tests

Prerequisites for integration tests:

1.  AWS account credentials in the environment.
2.  IAM role named `SageMakerRole` with `AmazonSageMakerFullAccess` and Elastic Inference permissions.
3.  Dummy ECR repo created: `aws ecr create-repository --repository-name remote-function-dummy-container`.

Run specific integration tests:

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

### Git Hooks

Enable git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

Or, enable an individual hook by moving it from `.githooks/` to `.git/hooks/`.

## Building Sphinx Docs

1.  Set up a Python environment and install dependencies:

    ```bash
    # conda
    conda create -n sagemaker python=3.12
    conda activate sagemaker
    conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

    # pip
    pip install -r doc/requirements.txt
    ```

2.  Clone/fork the repo and install the local version:

    ```bash
    pip install --upgrade .
    ```

3.  Build the docs:

    ```bash
    cd sagemaker-python-sdk/doc
    make html
    ```

4.  Preview the site:

    ```bash
    cd _build/html
    python -m http.server 8000
    ```

    Then go to [http://localhost:8000](http://localhost:8000) in your browser.

## SageMaker SparkML Serving

With SageMaker SparkML Serving, you can perform predictions against a SparkML Model in SageMaker. The model must be serialized with the `MLeap` library.

Supported Spark version: 3.3 (MLeap version - 0.20.0)

Example:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

Invoke the endpoint with a CSV payload:

```python
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

See the [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container) for more information.