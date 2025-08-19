<!-- Banner for SEO -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png">
  <img alt="SageMaker Banner" src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" height="100">
</picture>

# SageMaker Python SDK: Train, Deploy, and Manage Machine Learning Models

**Simplify and accelerate your machine learning workflows on Amazon SageMaker with the powerful and versatile SageMaker Python SDK.**  This SDK provides a user-friendly interface for training, deploying, and managing your machine learning models.  [Learn more on GitHub](https://github.com/aws/sagemaker-python-sdk).

[![PyPI Version](https://img.shields.io/pypi/v/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Conda-Forge Version](https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg)](https://anaconda.org/conda-forge/sagemaker-python-sdk)
[![Python Versions](https://img.shields.io/pypi/pyversions/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)
[![Documentation Status](https://readthedocs.org/projects/sagemaker/badge/?version=stable)](https://sagemaker.readthedocs.io/en/stable/)
[![CI Health](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg)](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml)


## Key Features:

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like **Apache MXNet** and **TensorFlow**.
*   **Built-in Algorithms:** Utilize pre-built, scalable, and optimized **Amazon algorithms** for core machine learning tasks.
*   **Custom Algorithms:** Bring your own algorithms packaged in SageMaker compatible Docker containers.
*   **Model Deployment:** Easily deploy your trained models for real-time inference.
*   **Model Tuning:** Leverage SageMaker's automatic model tuning capabilities.
*   **Batch Transform:** Perform batch predictions efficiently.
*   **Secure Training & Inference:** Train and deploy models within a Virtual Private Cloud (VPC) for enhanced security.
*   **SparkML Serving:** Deploy and perform predictions with SparkML models using the `MLeap` library.
*   **Model Monitoring:** Monitor model performance in production.
*   **Debugging:** Utilize the SageMaker Debugger for model troubleshooting.
*   **Processing:** Perform data preprocessing, feature engineering, and model evaluation.

## Installation

Install the SageMaker Python SDK using `pip`:

```bash
pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>
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

The `sagemaker` library has telemetry enabled to help understand user needs, diagnose issues, and deliver new features. This telemetry tracks the usage of various SageMaker functions.

If you prefer to opt out of telemetry, you can easily do so by setting the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration. For detailed instructions, please visit [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk).

## AWS Permissions

Amazon SageMaker operates on your behalf, requiring specific permissions. For more details on necessary permissions, refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  See the full license: [http://aws.amazon.com/apache2.0/](http://aws.amazon.com/apache2.0/)

## Running Tests

Install the test dependencies:

```bash
pip install --upgrade .[test]
```
or
```bash
pip install --upgrade .\[test\]
```

**Unit Tests:**

Run unit tests using tox:

```bash
tox tests/unit
```

**Integration Tests:**

Prerequisites:

1.  AWS account credentials are available in the environment for the boto3 client to use.
2.  An IAM role named `SageMakerRole` with the AmazonSageMakerFullAccess policy and permissions to use Elastic Inference.
3.  (For remote_function tests) A dummy ECR repo, created with: `aws ecr create-repository --repository-name remote-function-dummy-container`

Run integration tests (selectively):

```bash
tox -- -k 'test_i_care_about'
```
Run all integration tests (sequentially):

```bash
tox -- tests/integ
```
Run all integration tests (in parallel):

```bash
tox -- -n auto tests/integ
```

## Git Hooks

Enable all git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

Enable a single git hook:

```bash
mv .githooks/your_hook .git/hooks/
```

## Building Documentation

1.  **Setup Environment:**

    ```bash
    # conda
    conda create -n sagemaker python=3.12
    conda activate sagemaker
    conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

    # pip
    pip install -r doc/requirements.txt
    ```

2.  **Install Local Version:**

    ```bash
    pip install --upgrade .
    ```

3.  **Build Docs:**

    ```bash
    cd sagemaker-python-sdk/doc
    make html
    ```

4.  **Preview:**

    ```bash
    cd _build/html
    python -m http.server 8000
    ```

    Access at `http://localhost:8000`

## SageMaker SparkML Serving

Deploy and perform predictions against SparkML models serialized with the `MLeap` library.

**Supported Spark Version:** 3.3 (MLeap version - 0.20.0)

**Example:**

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

See [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container) for content-type and schema details.