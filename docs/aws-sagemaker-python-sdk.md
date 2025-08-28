![SageMaker Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models

**The Amazon SageMaker Python SDK simplifies the process of building, training, and deploying machine learning models on Amazon SageMaker.** ([View the original repository](https://github.com/aws/sagemaker-python-sdk))

The SageMaker Python SDK is your gateway to creating, training, and deploying machine learning models on Amazon SageMaker. This open-source library streamlines your ML workflow, offering flexibility and scalability for your projects.

## Key Features

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Built-in Algorithms:** Utilize scalable, optimized Amazon algorithms designed for SageMaker and GPU training.
*   **Custom Algorithm Support:** Train and host models with your own algorithms packaged in SageMaker-compatible Docker containers.
*   **SparkML Serving**: Deploy and perform predictions against a SparkML Model in SageMaker with MLeap Library.

## Getting Started

### Installation

Install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

## Core Functionality

The SDK provides tools for the following:

*   **Training:** Train models using various frameworks and algorithms.
*   **Deployment:** Deploy trained models for real-time or batch inference.
*   **Model Management:** Manage and track model versions and artifacts.
*   **Processing:** Run data preprocessing, feature engineering, and model evaluation jobs.

## Key Functionality Highlights

*   **SageMaker SparkML Serving**
*   **Automatic Model Tuning**
*   **Batch Transform**
*   **Inference Pipelines**
*   **Model Monitoring**
*   **Debugger**
*   **Processing**

## Documentation and Resources

For detailed documentation, including API reference and comprehensive guides, please visit:

*   [Read the Docs](https://sagemaker.readthedocs.io/)
*   [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk)
*   [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html)
*   [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container)
*   [MLeap](https://github.com/combust/mleap)

## Technical Details

*   **Supported Operating Systems:** Unix/Linux and Mac.
*   **Supported Python Versions:** Python 3.9, 3.10, 3.11, and 3.12.
*   **Telemetry:**  The ``sagemaker`` library has telemetry enabled to help us better understand user needs, diagnose issues, and deliver new features. You can opt out by setting the ``TelemetryOptOut`` parameter to ``true`` in the SDK defaults configuration.
*   **AWS Permissions:**  Requires appropriate AWS permissions for SageMaker operations.
*   **License:** Apache 2.0.

## Testing

### Running tests

You can install the libraries needed to run the tests by running :code:`pip install --upgrade .[test]` or, for Zsh users: :code:`pip install --upgrade .\[test\]`

**Unit tests**

To run the unit tests with tox, run:

```bash
tox tests/unit
```

**Integration tests**

To run the integration tests, the following prerequisites must be met

1. AWS account credentials are available in the environment for the boto3 client to use.
2. The AWS account has an IAM role named :code:`SageMakerRole`.
   It should have the AmazonSageMakerFullAccess policy attached as well as a policy with `the necessary permissions to use Elastic Inference <https://docs.aws.amazon.com/sagemaker/latest/dg/ei-setup.html>`__.
3. To run remote_function tests, dummy ecr repo should be created. It can be created by running -
    :code:`aws ecr create-repository --repository-name remote-function-dummy-container`

To run all of the integration tests in sequence:

```bash
tox -- tests/integ
```

To run the integration tests in parallel:

```bash
tox -- -n auto tests/integ
```

### Git Hooks

To enable all git hooks in the .githooks directory, run these commands in the repository directory:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

To enable an individual git hook, simply move it from the .githooks/ directory to the .git/hooks/ directory.

### Building Sphinx docs

Setup a Python environment, and install the dependencies listed in ``doc/requirements.txt``:

```bash
# conda
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

# pip
pip install -r doc/requirements.txt
```

Clone/fork the repo, and install your local version:

```bash
pip install --upgrade .
```

Then ``cd`` into the ``sagemaker-python-sdk/doc`` directory and run:

```bash
make html
```

You can edit the templates for any of the pages in the docs by editing the .rst files in the ``doc`` directory and then running ``make html`` again.

Preview the site with a Python web server:

```bash
cd _build/html
python -m http.server 8000
```

View the website by visiting http://localhost:8000