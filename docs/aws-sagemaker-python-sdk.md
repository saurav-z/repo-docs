<!-- Banner Image -->
<img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">

# Amazon SageMaker Python SDK

**Easily train and deploy your machine learning models on Amazon SageMaker with the open-source SageMaker Python SDK.**

This document provides an overview of the SageMaker Python SDK and its key features. For the complete documentation, including the API reference, visit [SageMaker Documentation](https://sagemaker.readthedocs.io/en/stable/).  You can also find the source code on [GitHub](https://github.com/aws/sagemaker-python-sdk).

## Key Features

*   **Simplified Model Training:** Train models using popular deep learning frameworks (Apache MXNet, TensorFlow, PyTorch, etc.), Amazon algorithms, and your own custom algorithms packaged in Docker containers.
*   **Flexible Deployment:** Deploy trained models for real-time inference or batch transform on SageMaker.
*   **Automated Model Tuning:** Leverage SageMaker's automatic model tuning capabilities to optimize your models.
*   **Model Monitoring:** Monitor model performance and detect data drift.
*   **Support for various ML frameworks**: Scikit-learn, XGBoost, Chainer

## Getting Started

### Installation

Install the latest version of the SageMaker Python SDK using `pip`:

```bash
pip install sagemaker
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Operating Systems

The SDK supports Unix/Linux and Mac operating systems.

### Supported Python Versions

The SDK is tested on:

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

##  Usage Guides

*   **Using the SageMaker Python SDK**: Access the core documentation.
*   **Using MXNet**: Leverage MXNet.
*   **Using TensorFlow**: Use TensorFlow.
*   **Using Chainer**: Implement Chainer.
*   **Using PyTorch**: Apply PyTorch.
*   **Using Scikit-learn**: Utilize Scikit-learn.
*   **Using XGBoost**: Incorporate XGBoost.
*   **SageMaker Reinforcement Learning Estimators**: Explore RL estimators.
*   **SageMaker SparkML Serving**: Perform predictions against a SparkML Model in SageMaker.
*   **Amazon SageMaker Built-in Algorithm Estimators**: Leverage the built-in algorithm estimators.
*   **Using SageMaker AlgorithmEstimators**: Access other documentation on algorithm estimators.
*   **Consuming SageMaker Model Packages**: Access documentation on Model Packages.
*   **BYO Docker Containers with SageMaker Estimators**: Bring your own docker containers.
*   **SageMaker Automatic Model Tuning**: Explore the automatic model tuning feature.
*   **SageMaker Batch Transform**: Leverage batch transform.
*   **Secure Training and Inference with VPC**: Learn about VPC security.
*   **BYO Model**: Use your own models.
*   **Inference Pipelines**: Access pipeline information.
*   **Amazon SageMaker Operators in Apache Airflow**:  Learn about operators in Airflow.
*   **SageMaker Autopilot**: Access Autopilot documentation.
*   **Model Monitoring**: Explore model monitoring features.
*   **SageMaker Debugger**: Access debugger documentation.
*   **SageMaker Processing**: Explore the processing feature.

## Telemetry

The SDK has telemetry enabled to help improve the product. You can opt out by setting the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration. See the [documentation](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for details.

## AWS Permissions

Amazon SageMaker performs operations on your behalf, so you need to have the required permissions. Please refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details.

## Licensing

The SageMaker Python SDK is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).

## Running Tests

Tests are available for unit tests and integration tests.

### Unit Tests

Run unit tests using `tox`:

```bash
tox tests/unit
```

### Integration Tests

Before running integration tests, ensure the following:

1.  AWS account credentials are available.
2.  An IAM role named `SageMakerRole` exists with the necessary permissions.
3.  A dummy ECR repo should be created. It can be created by running - :code:`aws ecr create-repository --repository-name remote-function-dummy-container`

Run integration tests with:

```bash
tox -- tests/integ
```

or to run in parallel:

```bash
tox -- -n auto tests/integ
```

## Git Hooks

To enable git hooks, run:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

## Building Sphinx Docs

Install dependencies from `doc/requirements.txt`, then install the local version of the SDK:

```bash
# conda
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

# pip
pip install -r doc/requirements.txt
pip install --upgrade .
```

Then, build the docs:

```bash
cd doc
make html
```

View the docs locally:

```bash
cd _build/html
python -m http.server 8000