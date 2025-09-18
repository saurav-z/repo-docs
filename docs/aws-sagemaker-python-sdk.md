<div align="center">
  <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">
  <h1>SageMaker Python SDK</h1>
  <p><b>Build, train, and deploy machine learning models on Amazon SageMaker with ease!</b></p>
  <a href="https://github.com/aws/sagemaker-python-sdk">
    <img src="https://img.shields.io/github/stars/aws/sagemaker-python-sdk?style=social" alt="Stars">
  </a>
</div>

---

The SageMaker Python SDK is your go-to open-source library for simplifying the end-to-end machine learning lifecycle on Amazon SageMaker. From model training and deployment to monitoring and management, this SDK empowers you to build, train, and deploy your models efficiently and effectively.

**Key Features:**

*   **Framework Agnostic:** Seamlessly train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Built-in Algorithms:** Leverage scalable Amazon algorithms optimized for SageMaker and GPU training.
*   **Bring Your Own Algorithms:** Easily integrate custom algorithms built into SageMaker-compatible Docker containers.
*   **Model Deployment:** Effortlessly deploy trained models with a few lines of code.
*   **Integration with MLeap:** Deploy SparkML models using the MLeap library for prediction against a SparkML Model in SageMaker.
*   **Model Monitoring:** Tools for monitoring model performance and identifying issues.
*   **Extensive Documentation:** Comprehensive documentation and API references available on [Read the Docs](https://sagemaker.readthedocs.io/).

---

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

Or install from source:

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

The `sagemaker` library includes telemetry to help us improve and deliver new features. You can opt-out by setting the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration. For more details, see [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk).

## AWS Permissions

Amazon SageMaker requires specific AWS permissions to operate on your behalf.  Consult the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details. If you use an IAM role with a path, grant permission for `iam:GetRole`.

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.  See [http://aws.amazon.com/apache2.0/](http://aws.amazon.com/apache2.0/) for details.

## Running Tests

Install the test dependencies:

```bash
pip install --upgrade .[test]
```

### Unit Tests

Run unit tests with tox:

```bash
tox tests/unit
```

### Integration Tests

Prerequisites:

1.  AWS credentials in the environment.
2.  An IAM role named `SageMakerRole`.
3.  A policy with the necessary permissions to use Elastic Inference.
4.  A dummy ECR repo named `remote-function-dummy-container`

Run integration tests (selectively):

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

Enable Git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

Enable an individual hook by moving it from `.githooks/` to `.git/hooks/`.

## Building Sphinx Docs

Install dependencies:

```bash
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

pip install -r doc/requirements.txt
```

Build the documentation:

```bash
pip install --upgrade .
cd sagemaker-python-sdk/doc
make html
```

Preview the docs:

```bash
cd _build/html
python -m http.server 8000
```

View the website by visiting http://localhost:8000

## SageMaker SparkML Serving

With SageMaker SparkML Serving, you can perform predictions against SparkML models serialized with the MLeap library.

Supported Spark version: 3.3 (MLeap version - 0.20.0)

Example:

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

See the [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container) for details on content types, accept formats, and the `schema`.