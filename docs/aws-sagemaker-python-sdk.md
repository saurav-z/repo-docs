# SageMaker Python SDK: Train and Deploy Machine Learning Models Easily

**Unlock the power of Amazon SageMaker with the SageMaker Python SDK, a comprehensive open-source library for streamlined model training and deployment.**  [View the original repo](https://github.com/aws/sagemaker-python-sdk)

This SDK simplifies the machine learning lifecycle, offering a user-friendly interface to build, train, and deploy models on Amazon SageMaker.

## Key Features:

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Leverage scalable, optimized implementations of core machine learning algorithms.
*   **Custom Algorithm Support:**  Bring your own algorithms packaged in SageMaker-compatible Docker containers.
*   **Flexible Deployment:** Deploy models using various instance types and configurations.
*   **Model Tuning:** Utilize SageMaker's automatic model tuning capabilities for optimal performance.
*   **Batch Transform:** Process large datasets efficiently with batch transform jobs.
*   **Comprehensive Documentation:** Access detailed API reference and usage guides at `Read the Docs <https://sagemaker.readthedocs.io>`_.
*   **SparkML Serving:**  Perform predictions against SparkML Models with MLeap library.

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

Install the latest version from PyPI:

```bash
pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>
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

### Telemetry

The ``sagemaker`` library has telemetry enabled to help us better understand user needs, diagnose issues, and deliver new features. This telemetry tracks the usage of various SageMaker functions.

If you prefer to opt out of telemetry, you can easily do so by setting the ``TelemetryOptOut`` parameter to ``true`` in the SDK defaults configuration. For detailed instructions, please visit `Configuring and using defaults with the SageMaker Python SDK <https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk>`__.

### AWS Permissions

Amazon SageMaker performs operations on your behalf.  Read more about necessary permissions in the `AWS Documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`__.

The SageMaker Python SDK should not require any additional permissions aside from what is required for using SageMaker.
However, if you are using an IAM role with a path in it, you should grant permission for ``iam:GetRole``.

### Licensing

Licensed under the Apache 2.0 License.  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  See: http://aws.amazon.com/apache2.0/

### Running Tests

Install testing dependencies:
```bash
pip install --upgrade .[test]
```
or
```bash
pip install --upgrade .\[test\]
```

**Unit tests**
Run unit tests with tox:
```bash
tox tests/unit
```

**Integration tests**

Prerequisites:
1.  AWS account credentials in the environment.
2.  IAM role named `SageMakerRole` with required permissions.
3.  Dummy ECR repo created with  `aws ecr create-repository --repository-name remote-function-dummy-container` (needed for remote_function tests)

Run a specific integration test:

```bash
tox -- -k 'test_i_care_about'
```

Run all integration tests sequentially (may take a while):

```bash
tox -- tests/integ
```

Run integration tests in parallel:
```bash
tox -- -n auto tests/integ
```

### Git Hooks

To enable all git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

Enable an individual hook by moving it from `.githooks/` to `.git/hooks/`.

### Building Sphinx docs

Setup a Python environment and install the dependencies from `doc/requirements.txt`:
```bash
# conda
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

# pip
pip install -r doc/requirements.txt
```

Then:

```bash
pip install --upgrade .
cd sagemaker-python-sdk/doc
make html
```

Preview the site:

```bash
cd _build/html
python -m http.server 8000
```

## SageMaker SparkML Serving

Deploy SparkML models serialized with MLeap.

Supported Spark version: 3.3 (MLeap version - 0.20.0)

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

Invoke the endpoint:

```python
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For details on `content-type`, `Accept` formats, and `schema`, see `SageMaker SparkML Serving Container`_.

.. _SageMaker SparkML Serving Container: https://github.com/aws/sagemaker-sparkml-serving-container