[![SageMaker Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)](https://github.com/aws/sagemaker-python-sdk)

# Amazon SageMaker Python SDK: Train, Deploy, and Manage Machine Learning Models

**The Amazon SageMaker Python SDK simplifies the entire machine learning workflow, empowering data scientists and developers to build, train, and deploy models with ease.** [Access the original repository here](https://github.com/aws/sagemaker-python-sdk).

Key Features:

*   **Framework Flexibility:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Built-in Algorithm Support:** Leverage scalable, optimized Amazon algorithms for core machine learning tasks.
*   **Bring Your Own Algorithms (BYOA):** Seamlessly integrate your custom algorithms within SageMaker compatible Docker containers.
*   **Model Deployment:** Easily deploy models for real-time inference.
*   **SparkML Serving:** Perform predictions against SparkML Models in SageMaker using MLeap.
*   **Integration and Automation:** Includes integration with tools like Apache Airflow.
*   **Model Monitoring:** Monitor and analyze the performance of your deployed models.
*   **Automated Machine Learning (AutoML):** Features Autopilot for automated model building.
*   **Model Debugging:** Utilize the SageMaker Debugger to identify and resolve issues in your models.

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

Install the latest version using pip:

```bash
pip install sagemaker
```

Or install from source:

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

The ``sagemaker`` library includes telemetry to understand user needs, diagnose issues, and provide new features.  You can opt-out by setting ``TelemetryOptOut`` to ``true`` in your SDK defaults configuration. For detailed instructions, see [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk).

## AWS Permissions

Amazon SageMaker requires specific permissions to perform operations on your behalf. Learn more about the necessary permissions in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html). The SageMaker Python SDK typically doesn't need additional permissions beyond those required for SageMaker usage, although, if you are using an IAM role with a path, ensure you have permission for ``iam:GetRole``.

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.  See [http://aws.amazon.com/apache2.0/](http://aws.amazon.com/apache2.0/) for details.

## Running Tests

To run the tests, install the testing dependencies:

```bash
pip install --upgrade .[test]
```

or, for Zsh users:

```bash
pip install --upgrade .\[test]
```

### Unit Tests

Run unit tests with tox:

```bash
tox tests/unit
```

### Integration Tests

Before running the integration tests, ensure the following prerequisites are met:

1.  AWS account credentials are available.
2.  An IAM role named `SageMakerRole` exists in your AWS account, with `AmazonSageMakerFullAccess` policy attached and permissions for Elastic Inference (if applicable).
3.  For remote_function tests, create a dummy ECR repository:
    ```bash
    aws ecr create-repository --repository-name remote-function-dummy-container
    ```

Run selective integration tests:

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

## Git Hooks

To enable all git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

To enable a single git hook, move it from the `.githooks/` directory to the `.git/hooks/` directory.

## Building Sphinx Docs

1.  Set up a Python environment and install dependencies from `doc/requirements.txt`:

    ```bash
    # conda
    conda create -n sagemaker python=3.12
    conda activate sagemaker
    conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

    # pip
    pip install -r doc/requirements.txt
    ```

2.  Install your local version of the SDK:

    ```bash
    pip install --upgrade .
    ```

3.  Build the HTML docs:

    ```bash
    cd sagemaker-python-sdk/doc
    make html
    ```

4.  Preview the site:

    ```bash
    cd _build/html
    python -m http.server 8000
    ```

    Then, view the website at `http://localhost:8000`.

## SageMaker SparkML Serving

SageMaker SparkML Serving enables you to perform predictions against SparkML Models in SageMaker.  Models must be serialized with the `MLeap` library.

*   **Supported Spark Version:** 3.3 (MLeap version - 0.20.0)

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

For more details on content types, accepted formats, and schema structure, see the [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).