[![SageMaker Python SDK Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)](https://github.com/aws/sagemaker-python-sdk)

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models

**Simplify your machine learning workflow with the Amazon SageMaker Python SDK, a powerful open-source library for building, training, and deploying models on Amazon SageMaker.** ([See the original repo](https://github.com/aws/sagemaker-python-sdk))

## Key Features:

*   **Framework Support:** Train and deploy models using popular deep learning frameworks, including Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Utilize scalable, pre-built machine learning algorithms optimized for SageMaker.
*   **Custom Algorithms:** Bring your own algorithms packaged in SageMaker-compatible Docker containers.
*   **Simplified Deployment:** Easily deploy your trained models for real-time or batch inference.
*   **Integration:** Seamlessly integrates with other SageMaker features like automatic model tuning, batch transform, and model monitoring.
*   **Broad Compatibility:** Supports Unix/Linux, Mac, and multiple Python versions (3.9, 3.10, 3.11, and 3.12).

## Table of Contents

*   [Installing SageMaker Python SDK](#installing-the-sagemaker-python-sdk)
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

You can easily install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker==<Latest version from pypi from https://pypi.org/project/sagemaker/>
```

Or, to install from source:

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

The `sagemaker` library uses telemetry to understand user needs and improve the product. You can opt-out by setting the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration.  See [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for details.

### AWS Permissions

The SageMaker Python SDK requires the necessary AWS permissions to interact with SageMaker resources.  Refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for information on required permissions.  If using an IAM role with a path, you may also need to grant permission for `iam:GetRole`.

### Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. The license is available at:  [http://aws.amazon.com/apache2.0/](http://aws.amazon.com/apache2.0/)

## Running tests

### Unit tests

Install test dependencies:

```bash
pip install --upgrade .[test]
```
or for Zsh users:
```bash
pip install --upgrade .\[test\]
```

Run unit tests with tox:

```bash
tox tests/unit
```

### Integration tests

Prerequisites:

1.  AWS account credentials must be available in the environment for the boto3 client.
2.  An IAM role named `SageMakerRole` with the AmazonSageMakerFullAccess policy and necessary permissions to use Elastic Inference (see [Elastic Inference setup](https://docs.aws.amazon.com/sagemaker/latest/dg/ei-setup.html)) must be available in the account.
3.  For remote function tests, create a dummy ECR repo named `remote-function-dummy-container`
    ```bash
    aws ecr create-repository --repository-name remote-function-dummy-container
    ```

Run integration tests:

*   Run a specific test:

    ```bash
    tox -- -k 'test_i_care_about'
    ```

*   Run all integration tests sequentially:

    ```bash
    tox -- tests/integ
    ```

*   Run all integration tests in parallel:

    ```bash
    tox -- -n auto tests/integ
    ```

### Git Hooks

Enable all git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

Enable an individual git hook: Move the hook from `.githooks/` to `.git/hooks/`.

## Building Sphinx docs

Install dependencies (create a conda environment is recommended):

```bash
# conda
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

# pip
pip install -r doc/requirements.txt
```

Install your local version:

```bash
pip install --upgrade .
```

Build the documentation:

```bash
cd sagemaker-python-sdk/doc
make html
```

View the documentation:

```bash
cd _build/html
python -m http.server 8000
```

Then visit http://localhost:8000

## SageMaker SparkML Serving

Utilize SparkML models within SageMaker for serving predictions.

Supported Spark version: 3.3 (MLeap version - 0.20.0)

Example:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

Invoke endpoint:

```python
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

See the [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container) for details on content-type, accept formats, and schema structure.