<div align="center">
  <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">
</div>

# SageMaker Python SDK: Train and Deploy Machine Learning Models on AWS

**Empower your machine learning workflows on Amazon SageMaker with the SageMaker Python SDK, enabling efficient training and deployment of models.**

**[Explore the SageMaker Python SDK on GitHub](https://github.com/aws/sagemaker-python-sdk)**

## Key Features

*   **Simplified Model Training:** Easily train models using popular deep learning frameworks like Apache MXNet and TensorFlow, or leverage pre-built Amazon algorithms.
*   **Flexible Deployment:** Deploy models with Docker containers, enabling you to bring your own algorithms to SageMaker.
*   **Comprehensive Framework Support:** Supports a variety of frameworks, including Apache MXNet, TensorFlow, PyTorch, Scikit-learn, XGBoost and Chainer.
*   **Built-in Algorithms:** Access scalable implementations of core machine learning algorithms optimized for SageMaker and GPU training.
*   **SparkML Serving:** Deploy and perform predictions against SparkML Models using the MLeap library
*   **Integration with SageMaker Features:** Seamlessly integrate with advanced SageMaker features like Automatic Model Tuning, Batch Transform, and Model Monitoring.

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

Install the latest version of the SageMaker Python SDK using pip:

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

## Telemetry

The `sagemaker` library has telemetry enabled to help us better understand user needs, diagnose issues, and deliver new features. This telemetry tracks the usage of various SageMaker functions.

If you prefer to opt out of telemetry, you can easily do so by setting the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration. For detailed instructions, please visit [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk).

## AWS Permissions

Amazon SageMaker, as a managed service, operates on your behalf and can perform only operations that the user permits. You can read more about which permissions are necessary in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

The SageMaker Python SDK does not require any additional permissions besides what is required for using SageMaker. However, if you are using an IAM role with a path in it, you should grant permission for `iam:GetRole`.

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License. It is copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. The license is available at: [http://aws.amazon.com/apache2.0/](http://aws.amazon.com/apache2.0/)

## Running tests

SageMaker Python SDK has unit tests and integration tests.

You can install the libraries needed to run the tests by running :code:`pip install --upgrade .[test]` or, for Zsh users: :code:`pip install --upgrade .\[test\]`

### Unit tests

We run unit tests with tox, which is a program that lets you run unit tests for multiple Python versions, and also make sure the code fits our style guidelines. We run tox with `all of our supported Python versions <#supported-python-versions>`_, so to run unit tests with the same configuration we do, you need to have interpreters for those Python versions installed.

To run the unit tests with tox, run:

```bash
tox tests/unit
```

### Integration tests

To run the integration tests, the following prerequisites must be met

1.  AWS account credentials are available in the environment for the boto3 client to use.
2.  The AWS account has an IAM role named :code:`SageMakerRole`. It should have the AmazonSageMakerFullAccess policy attached as well as a policy with `the necessary permissions to use Elastic Inference <https://docs.aws.amazon.com/sagemaker/latest/dg/ei-setup.html>`__.
3.  To run remote_function tests, dummy ecr repo should be created. It can be created by running -
    :code:`aws ecr create-repository --repository-name remote-function-dummy-container`

We recommend selectively running just those integration tests you'd like to run. You can filter by individual test function names with:

```bash
tox -- -k 'test_i_care_about'
```

You can also run all of the integration tests by running the following command, which runs them in sequence, which may take a while:

```bash
tox -- tests/integ
```

You can also run them in parallel:

```bash
tox -- -n auto tests/integ
```

## Git Hooks

To enable all git hooks in the .githooks directory, run these commands in the repository directory:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

To enable an individual git hook, simply move it from the .githooks/ directory to the .git/hooks/ directory.

## Building Sphinx docs

Setup a Python environment, and install the dependencies listed in ``doc/requirements.txt``:

```bash
# conda
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

# pip
pip install -r doc/requirements.txt
```

Clone/fork the repo, and install your local version:

```bash
pip install --upgrade .
```

Then ``cd`` into the ``sagemaker-python-sdk/doc`` directory and run:

```bash
make html
```

You can edit the templates for any of the pages in the docs by editing the .rst files in the ``doc`` directory and then running ``make html`` again.

Preview the site with a Python web server:

```bash
cd _build/html
python -m http.server 8000
```

View the website by visiting http://localhost:8000

## SageMaker SparkML Serving

With SageMaker SparkML Serving, you can now perform predictions against a SparkML Model in SageMaker. In order to host a SparkML model in SageMaker, it should be serialized with the ``MLeap`` library.

For more information on MLeap, see https://github.com/combust/mleap .

Supported major version of Spark: 3.3 (MLeap version - 0.20.0)

Here is an example on how to create an instance of  ``SparkMLModel`` class and use ``deploy()`` method to create an endpoint which can be used to perform prediction against your trained SparkML Model.

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

Once the model is deployed, we can invoke the endpoint with a ``CSV`` payload like this:

```python
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more information about the different ``content-type`` and ``Accept`` formats as well as the structure of the ``schema`` that SageMaker SparkML Serving recognizes, please see [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).