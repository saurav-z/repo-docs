<!-- SageMaker Python SDK Banner -->
<p align="center">
  <a href="https://github.com/aws/sagemaker-python-sdk">
    <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">
  </a>
</p>

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models

**The Amazon SageMaker Python SDK simplifies the end-to-end machine learning workflow, enabling you to build, train, and deploy models efficiently on Amazon SageMaker.**

[![PyPI Version](https://img.shields.io/pypi/v/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Conda-Forge Version](https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg)](https://anaconda.org/conda-forge/sagemaker-python-sdk)
[![Python Versions](https://img.shields.io/pypi/pyversions/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)
[![Documentation Status](https://readthedocs.org/projects/sagemaker/badge/?version=stable)](https://sagemaker.readthedocs.io/en/stable/)
[![CI Health](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg)](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml)

## Key Features of the SageMaker Python SDK

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Built-in Algorithms:** Utilize scalable Amazon algorithms optimized for SageMaker and GPU training.
*   **Custom Algorithms:** Bring your own algorithms built into SageMaker compatible Docker containers.
*   **Easy Model Deployment:** Deploy your trained models with just a few lines of code.
*   **Integration with SageMaker Services:** Leverage other SageMaker features like Automatic Model Tuning, Batch Transform, Model Monitoring, and more.
*   **SparkML Serving:** Deploy and perform predictions against SparkML models using the MLeap library.

## Getting Started

### Installation

Install the latest version using pip:

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

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

## Documentation

For comprehensive documentation, including API references and usage examples, please visit the [SageMaker Documentation](https://sagemaker.readthedocs.io/en/stable/).

## Sections

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

## Telemetry

The `sagemaker` library has telemetry enabled to help us understand user needs and diagnose issues. You can opt out by setting the `TelemetryOptOut` parameter to `true`.

## AWS Permissions

Amazon SageMaker performs operations on your behalf, requiring specific permissions.  See the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details.  Grant permission for `iam:GetRole` if using an IAM role with a path.

## Licensing

This project is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/). Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

## Testing

### Unit Tests

Run unit tests using tox:

```bash
tox tests/unit
```

### Integration Tests

Run integration tests with:

```bash
tox -- tests/integ
```

Or, selectively run tests:

```bash
tox -- -k 'test_i_care_about'
```
## Git Hooks

Enable git hooks with these commands:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

## Building Sphinx Docs

Follow these steps to build the documentation:

1.  **Set up a Python environment and install dependencies:**

    ```bash
    # conda
    conda create -n sagemaker python=3.12
    conda activate sagemaker
    conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

    # pip
    pip install -r doc/requirements.txt
    ```

2.  **Clone/fork the repository and install the local version:**

    ```bash
    pip install --upgrade .
    ```

3.  **Build the HTML documentation:**

    ```bash
    cd sagemaker-python-sdk/doc
    make html
    ```

4.  **Preview the site:**

    ```bash
    cd _build/html
    python -m http.server 8000
    ```

    Then, visit [http://localhost:8000](http://localhost:8000) in your browser.

## SageMaker SparkML Serving

Deploy and perform predictions against a SparkML model in SageMaker.

**Example:**

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

**Invoke endpoint:**

```python
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more information about the different `content-type` and `Accept` formats as well as the structure of the `schema` that SageMaker SparkML Serving recognizes, please see [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).

[Back to Top](#amazon-sagemaker-python-sdk-train-and-deploy-machine-learning-models)

**[Visit the original repository](https://github.com/aws/sagemaker-python-sdk) for more information and contributions.**