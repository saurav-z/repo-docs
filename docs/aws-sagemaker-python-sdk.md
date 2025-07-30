<!-- SageMaker Python SDK Banner -->
<img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">

# SageMaker Python SDK: Train and Deploy Machine Learning Models with Ease

**Quickly build, train, and deploy machine learning models on Amazon SageMaker with the user-friendly SageMaker Python SDK!** ([View the original repository](https://github.com/aws/sagemaker-python-sdk))

## Key Features:

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Utilize scalable, optimized Amazon algorithms for core machine learning tasks.
*   **Bring Your Own Algorithms:** Seamlessly integrate your custom algorithms built into SageMaker-compatible Docker containers.
*   **Model Deployment:** Easily deploy your trained models for real-time or batch inference.
*   **Automatic Model Tuning:** Optimize your models with SageMaker's automatic model tuning capabilities.
*   **Batch Transform:** Perform batch predictions on large datasets efficiently.
*   **Secure Training and Inference:** Train and deploy your models within a VPC for enhanced security.
*   **Integration with SparkML Serving:**  Easily serve SparkML models with the SageMaker SparkML Serving container.
*   **Model Monitoring:** Gain insights into your models' performance with built-in monitoring tools.
*   **And much more!**

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

The SageMaker Python SDK is available on PyPI.  Install the latest version with pip:

```bash
pip install sagemaker
```

You can also install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Operating Systems

*   Unix/Linux
*   Mac

### Supported Python Versions

The SDK is tested on the following Python versions:

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

### Telemetry

The `sagemaker` library has telemetry enabled to help us understand user needs and deliver new features. To opt-out:

```python
from sagemaker.config import set_defaults
set_defaults(TelemetryOptOut=True)
```

For details, see: [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk).

### AWS Permissions

Amazon SageMaker requires specific permissions to operate on your behalf. For detailed information, consult the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

The SDK generally doesn't require additional permissions beyond what's needed for SageMaker.  If you're using an IAM role with a path, ensure you have permission for `iam:GetRole`.

### Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.

### Running Tests

Install testing dependencies:

```bash
pip install --upgrade .[test]
# or for Zsh users:
pip install --upgrade .\[test\]
```

**Unit Tests:**

Run unit tests using tox:

```bash
tox tests/unit
```

**Integration Tests:**

Prerequisites:

1.  AWS account credentials.
2.  An IAM role named `SageMakerRole` with `AmazonSageMakerFullAccess` and necessary permissions for Elastic Inference.
3.  A dummy ECR repo for remote_function tests: `aws ecr create-repository --repository-name remote-function-dummy-container`

Run specific integration tests:

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

### Git Hooks

Enable Git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

Enable an individual hook by moving it from `.githooks/` to `.git/hooks/`.

### Building Sphinx Docs

1.  Setup a Python environment and install dependencies from `doc/requirements.txt`:

    ```bash
    # conda
    conda create -n sagemaker python=3.12
    conda activate sagemaker
    conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

    # pip
    pip install -r doc/requirements.txt
    ```

2.  Install your local version:

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

    Then, view the website at: `http://localhost:8000`

## SageMaker SparkML Serving

With SageMaker SparkML Serving, you can perform predictions against SparkML models in SageMaker.

To host a SparkML model, it must be serialized with the `MLeap` library.

Supported Spark version: 3.3 (MLeap version: 0.20.0)

Here's how to create a `SparkMLModel` instance and deploy it:

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

For details on `content-type`, `Accept` formats, and the `schema`, see the [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).