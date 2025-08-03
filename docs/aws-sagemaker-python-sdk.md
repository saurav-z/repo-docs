[![SageMaker](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)](https://github.com/aws/sagemaker-python-sdk)

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models

**The Amazon SageMaker Python SDK empowers data scientists and developers to build, train, and deploy machine learning models seamlessly on Amazon SageMaker.**

**Key Features:**

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Leverage scalable, optimized implementations of core machine learning algorithms.
*   **Bring Your Own Algorithms:** Train and host models using custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Model Deployment:** Deploy trained models with a few lines of code.
*   **Flexible Training:** Supports various training methods, including distributed training, hyperparameter tuning, and more.
*   **Model Monitoring & Debugging:** Integrate with SageMaker's Model Monitoring and Debugger tools for better model performance.
*   **SparkML Serving Integration:**  Easily deploy and perform predictions against SparkML models using MLeap library.

**Get started with the SageMaker Python SDK by visiting the [official GitHub repository](https://github.com/aws/sagemaker-python-sdk) to explore the extensive features and functionalities.**

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

The SageMaker Python SDK is available on PyPI. You can install the latest version using pip:

```bash
pip install sagemaker
```

You can also install from source by cloning the repository:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Operating Systems

The SDK supports Unix/Linux and macOS.

### Supported Python Versions

The SageMaker Python SDK is tested on:

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

### Telemetry

The `sagemaker` library has telemetry enabled to gather insights into user behavior, diagnose issues, and develop new features. This telemetry tracks the usage of various SageMaker functions.

To opt-out, set the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration.  For detailed instructions, see [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk).

### AWS Permissions

Amazon SageMaker, as a managed service, operates on your behalf using AWS hardware. It can only perform operations permitted by your user's permissions.  Review the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details on required permissions.

Generally, the SageMaker Python SDK does not require additional permissions beyond those needed for SageMaker usage.  However, if using an IAM role with a path, grant permission for `iam:GetRole`.

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.

*   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
*   License: [http://aws.amazon.com/apache2.0/](http://aws.amazon.com/apache2.0/)

## Running Tests

The SDK includes both unit and integration tests.

Install the test dependencies:

```bash
pip install --upgrade .[test]
```

or, for Zsh users:

```bash
pip install --upgrade .\[test\]
```

### Unit Tests

Run unit tests using `tox`. Ensure you have Python interpreters for the supported versions installed.

Run unit tests with:

```bash
tox tests/unit
```

### Integration Tests

Prerequisites for running integration tests:

1.  AWS account credentials configured for the `boto3` client.
2.  An IAM role named `SageMakerRole` with `AmazonSageMakerFullAccess` and permissions for Elastic Inference, as described in the AWS documentation.
3.  A dummy ECR repo (e.g., `remote-function-dummy-container`) for remote function tests.  Create with: `aws ecr create-repository --repository-name remote-function-dummy-container`

To run specific integration tests:

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

Enable Git hooks in the `.git/hooks` directory:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

Enable an individual hook by moving it from `.githooks/` to `.git/hooks/`.

## Building Sphinx Docs

1.  Set up a Python environment and install dependencies (from `doc/requirements.txt`):

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

3.  Navigate to the `doc` directory and run:

    ```bash
    cd sagemaker-python-sdk/doc
    make html
    ```

4.  Edit the `.rst` files in the `doc` directory to modify the documentation. Then run `make html` again.

5.  Preview the site:

    ```bash
    cd _build/html
    python -m http.server 8000
    ```

6.  View the site at [http://localhost:8000](http://localhost:8000)

## SageMaker SparkML Serving

With SageMaker SparkML Serving, you can perform predictions against SparkML models in SageMaker.  Models must be serialized using the `MLeap` library.

For details about MLeap, see [https://github.com/combust/mleap](https://github.com/combust/mleap).

Supported Spark version: 3.3 (with MLeap version 0.20.0)

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

For details on `content-type`, `Accept` formats, and the `schema`, see the [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container) documentation.