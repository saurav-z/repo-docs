<div align="center">
  <a href="https://github.com/aws/sagemaker-python-sdk">
    <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">
  </a>
  <h1>Amazon SageMaker Python SDK</h1>
  <p><b>Simplify your machine learning workflows on Amazon SageMaker with this powerful and versatile Python SDK.</b></p>
</div>

[![PyPI Version](https://img.shields.io/pypi/v/sagemaker.svg)](https://pypi.org/project/sagemaker)
[![Conda-Forge Version](https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg)](https://anaconda.org/conda-forge/sagemaker-python-sdk)
[![Python Versions](https://img.shields.io/pypi/pyversions/sagemaker.svg)](https://pypi.org/project/sagemaker)
[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)
[![Documentation Status](https://readthedocs.org/projects/sagemaker/badge/?version=stable)](https://sagemaker.readthedocs.io/en/stable/)
[![CI Health](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg)](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml)

The Amazon SageMaker Python SDK is an open-source library that streamlines the process of building, training, and deploying machine learning models on Amazon SageMaker. Whether you're a seasoned data scientist or just starting out, this SDK provides the tools you need to efficiently manage your ML projects.

**Key Features:**

*   **Framework Support:** Train and deploy models using popular deep learning frameworks such as Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Leverage scalable, optimized implementations of core machine learning algorithms.
*   **Custom Algorithms:**  Bring your own algorithms packaged in SageMaker compatible Docker containers.
*   **Model Deployment:** Easily deploy your trained models for real-time or batch predictions.
*   **Integration:** Seamlessly integrates with other AWS services.
*   **Flexible:** Supports various use cases from simple training to complex model pipelines.

**Table of Contents**

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
pip install sagemaker==<Latest version from pypi from https://pypi.org/project/sagemaker/>
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

### Telemetry

The ``sagemaker`` library includes telemetry to help us improve the SDK.  You can opt out by setting the ``TelemetryOptOut`` parameter to ``true`` in the SDK defaults configuration.  See [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for details.

### AWS Permissions

SageMaker requires specific permissions to operate.  Please refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for the necessary IAM role setup.  If you're using an IAM role with a path, grant permission for `iam:GetRole`.

### Licensing

The SageMaker Python SDK is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).

### Running Tests

Install test dependencies:

```bash
pip install --upgrade .[test]
```

or for Zsh users:

```bash
pip install --upgrade .\[test\]
```

**Unit Tests**

Run unit tests using tox:

```bash
tox tests/unit
```

**Integration Tests**

Prerequisites for integration tests:

1.  AWS account credentials in the environment.
2.  An IAM role named `SageMakerRole` with `AmazonSageMakerFullAccess` and necessary permissions for Elastic Inference (if applicable).
3.  A dummy ECR repo created for `remote_function` tests:  `aws ecr create-repository --repository-name remote-function-dummy-container`

Run specific integration tests:

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

Enable all git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

Enable a single hook: Move it from `.githooks/` to `.git/hooks/`.

### Building Sphinx Docs

1.  Set up a Python environment and install dependencies (using conda or pip):

    ```bash
    # conda
    conda create -n sagemaker python=3.12
    conda activate sagemaker
    conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

    # pip
    pip install -r doc/requirements.txt
    ```

2.  Install the local version:

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

    View the website by visiting http://localhost:8000

## SageMaker SparkML Serving

Integrate SparkML models with SageMaker for real-time predictions.  Models must be serialized with the MLeap library.

Supported Spark version: 3.3 (MLeap version - 0.20.0)

Example:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For details on `content-type`, `Accept` formats, and `schema`, refer to the [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).

[Back to Top](#amazon-sagemaker-python-sdk)