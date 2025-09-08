<div align="center">
  <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" width="300">
</div>

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models

**Unlock the power of Amazon SageMaker with the open-source Python SDK, streamlining your machine learning workflow from model training to deployment.**  ([View the Original Repository](https://github.com/aws/sagemaker-python-sdk))

[![PyPI Version](https://img.shields.io/pypi/v/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Conda-Forge Version](https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg)](https://anaconda.org/conda-forge/sagemaker-python-sdk)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)
[![Documentation Status](https://readthedocs.org/projects/sagemaker/badge/?version=stable)](https://sagemaker.readthedocs.io/en/stable/)
[![CI Health](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg)](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml)

## Key Features:

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet, TensorFlow, PyTorch, and more.
*   **Built-in Algorithms:** Leverage scalable, optimized Amazon algorithms for core machine learning tasks.
*   **Custom Algorithms:** Bring your own algorithms packaged in SageMaker-compatible Docker containers.
*   **Simplified Deployment:** Easily deploy models with a few lines of code.
*   **Model Hosting:** Host your trained models for real-time predictions.
*   **Integration:** Seamlessly integrate with Amazon SageMaker features like Automatic Model Tuning, Batch Transform, and Model Monitoring.
*   **SparkML Serving:** Deploy and serve SparkML models serialized with MLeap.

## Installation

Install the latest version via pip:

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

## AWS Permissions

The SageMaker Python SDK requires the permissions needed for using SageMaker.  For more details, refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).  If using an IAM role with a path, grant permission for `iam:GetRole`.

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License. Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. [License Details](http://aws.amazon.com/apache2.0/).

## Running Tests

Install testing dependencies:
```bash
pip install --upgrade .[test]
```

**Unit Tests:**
```bash
tox tests/unit
```

**Integration Tests:**
```bash
tox -- tests/integ
```

## Git Hooks

To enable git hooks:
```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

## Building Sphinx Documentation

1.  Set up a Python environment and install dependencies:

    ```bash
    # conda
    conda create -n sagemaker python=3.12
    conda activate sagemaker
    conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

    # pip
    pip install -r doc/requirements.txt
    ```

2.  Clone/Fork the repository and install the local version:

    ```bash
    pip install --upgrade .
    ```
3.  Build the documentation:
    ```bash
    cd sagemaker-python-sdk/doc
    make html
    ```

4.  Preview the documentation (optional):
    ```bash
    cd _build/html
    python -m http.server 8000
    ```
    View at http://localhost:8000.

## SageMaker SparkML Serving

Deploy SparkML models using MLeap.

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

For more information, see the [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).