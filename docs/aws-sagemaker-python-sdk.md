![SageMaker Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)

# Amazon SageMaker Python SDK: Simplify Machine Learning on AWS

**The Amazon SageMaker Python SDK empowers you to train and deploy machine learning models at scale using Python.**  This library simplifies your machine learning workflow, from model training to deployment and monitoring, all within the Amazon SageMaker environment.  Get started with the [original repo](https://github.com/aws/sagemaker-python-sdk).

Key Features:

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet, TensorFlow, PyTorch, and more.
*   **Built-in Algorithms:** Leverage scalable Amazon SageMaker algorithms optimized for efficient training on AWS.
*   **Custom Algorithm Support:** Bring your own algorithms packaged in SageMaker-compatible Docker containers.
*   **Simplified Deployment:** Easily deploy your models for real-time inference using the SageMaker platform.
*   **Model Monitoring:** Monitor model performance and drift to maintain accuracy and reliability.
*   **Extensive Functionality:** Utilize features like automatic model tuning, batch transform, and inference pipelines.

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

The SageMaker Python SDK is available on PyPI. Install the latest version using pip:

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

The `sagemaker` library includes telemetry to help improve user experience and diagnose issues.  You can opt out by setting the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration.  See [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for details.

### AWS Permissions

The SageMaker Python SDK requires appropriate AWS permissions for interacting with SageMaker.  Refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details. If using an IAM role with a path, grant permission for `iam:GetRole`.

### Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.  Copyright Amazon.com, Inc. or its affiliates.  See http://aws.amazon.com/apache2.0/

## Running Tests

To run tests:

1.  Install testing dependencies:
    ```bash
    pip install --upgrade .[test]
    ```
    or, for Zsh users:
    ```bash
    pip install --upgrade .\[test\]
    ```

2.  **Unit Tests**:  Run unit tests using tox. Ensure you have interpreters for supported Python versions installed.

    ```bash
    tox tests/unit
    ```

3.  **Integration Tests**:

    *   Ensure AWS account credentials are available.
    *   An IAM role named `SageMakerRole` with `AmazonSageMakerFullAccess` and the necessary permissions for Elastic Inference is required.
    *   To run remote_function tests, create a dummy ECR repo: `aws ecr create-repository --repository-name remote-function-dummy-container`

    Run individual tests:

    ```bash
    tox -- -k 'test_i_care_about'
    ```

    Run all integration tests (may take a while):

    ```bash
    tox -- tests/integ
    ```

    Run integration tests in parallel:

    ```bash
    tox -- -n auto tests/integ
    ```

## Git Hooks

Enable all git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

Enable an individual git hook by moving it from `.githooks/` to `.git/hooks/`.

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

2.  Clone/fork the repo and install your local version:

    ```bash
    pip install --upgrade .
    ```

3.  Build the documentation:

    ```bash
    cd sagemaker-python-sdk/doc
    make html
    ```

4.  Preview the site:

    ```bash
    cd _build/html
    python -m http.server 8000
    ```

    View at http://localhost:8000

## SageMaker SparkML Serving

Deploy and serve SparkML models using the SageMaker SparkML Serving container.  SparkML models must be serialized using the `MLeap` library.

*   Supported Spark version: 3.3 (MLeap version - 0.20.0)

Example:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more information, see the [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).