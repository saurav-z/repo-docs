# SageMaker Python SDK: Train & Deploy Machine Learning Models with Ease

**Unlock the power of Amazon SageMaker with the SageMaker Python SDK, a versatile open-source library for building, training, and deploying machine learning models.** Learn more and contribute on the original repo: [https://github.com/aws/sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk)

**Key Features:**

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Leverage scalable, optimized implementations of core machine learning algorithms.
*   **Custom Algorithm Support:** Train and host models using your own algorithms packaged in SageMaker-compatible Docker containers.
*   **Model Deployment:** Easily deploy your trained models for real-time or batch inference.
*   **SparkML Integration:** Seamlessly deploy and perform predictions against SparkML models serialized with MLeap.
*   **Model Tuning and Monitoring:** Utilize features like automatic model tuning, batch transform, and model monitoring for optimal performance.
*   **Secure Operations:** Supports secure training and inference within a Virtual Private Cloud (VPC).

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
pip install sagemaker
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

The SDK is tested on:

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

### Telemetry

The `sagemaker` library includes telemetry to help improve the SDK. You can opt out by setting the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration. See the [documentation](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for more details.

### AWS Permissions

The SageMaker Python SDK requires the necessary permissions to interact with Amazon SageMaker.  Refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details on required permissions. If using an IAM role with a path, grant permission for `iam:GetRole`.

### Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.  Copyright Amazon.com, Inc. or its affiliates.  The full license is available at: [http://aws.amazon.com/apache2.0/](http://aws.amazon.com/apache2.0/)

## Running Tests

The SDK includes unit and integration tests.

Install testing dependencies:

```bash
pip install --upgrade .[test]
```
Or, for Zsh users:
```bash
pip install --upgrade .\[test]
```

### Unit Tests

Run unit tests using tox:

```bash
tox tests/unit
```

### Integration Tests

**Prerequisites:**

1.  AWS account credentials configured.
2.  An IAM role named `SageMakerRole` with `AmazonSageMakerFullAccess` policy and necessary Elastic Inference permissions.
3.  A dummy ECR repository created for remote function tests:  `aws ecr create-repository --repository-name remote-function-dummy-container`

**Running Specific Tests:**

```bash
tox -- -k 'test_i_care_about'
```

**Running All Integration Tests (May take a while):**

```bash
tox -- tests/integ
```

**Running Integration Tests in Parallel:**

```bash
tox -- -n auto tests/integ
```

## Git Hooks

Enable git hooks from the `.githooks` directory:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

Or, to enable an individual git hook, move it from `.githooks/` to `.git/hooks/`.

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

2.  Clone/fork the repository and install your local version:

    ```bash
    pip install --upgrade .
    ```

3.  Build the HTML documentation:

    ```bash
    cd sagemaker-python-sdk/doc
    make html
    ```

4.  Preview the site:

    ```bash
    cd _build/html
    python -m http.server 8000
    ```

    Then navigate to `http://localhost:8000` in your browser.

## SageMaker SparkML Serving

Deploy and perform predictions against SparkML models using the `SparkMLModel` class. Models need to be serialized with the `MLeap` library.

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

For detailed information about content types, accepted formats, and the schema, see the [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).