<!-- Banner Image (Kept from original for visual appeal) -->
<img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">

# SageMaker Python SDK: Train and Deploy Machine Learning Models with Ease

**Unlock the power of Amazon SageMaker** with the open-source SageMaker Python SDK, a comprehensive library for building, training, and deploying machine learning models. This SDK simplifies the entire ML workflow, from data preparation to model deployment, all within the familiar Python environment.  ([View the Original Repo](https://github.com/aws/sagemaker-python-sdk))

<!-- Badges (Kept from original, important for project overview) -->
[![PyPI Version](https://img.shields.io/pypi/v/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Conda-Forge Version](https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg)](https://anaconda.org/conda-forge/sagemaker-python-sdk)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)
[![Documentation Status](https://readthedocs.org/projects/sagemaker/badge/?version=stable)](https://sagemaker.readthedocs.io/en/stable/)
[![CI Health](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg)](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml)

## Key Features

*   **Framework Agnostic:** Supports popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Built-in Algorithms:** Leverage scalable, optimized Amazon algorithms for common machine learning tasks.
*   **Bring Your Own Algorithms (BYO):** Easily integrate your custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Simplified Training and Deployment:** Streamline model training, tuning, and deployment processes.
*   **Model Monitoring:** Keep track of how your models are performing over time.
*   **Integration:** Works with other AWS services such as Apache Airflow, and SparkML.
*   **Model Serving:** Allows for performing predictions against a SparkML Model in SageMaker.

## Table of Contents

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

You can install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>
```

Or, install from source:

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

The `sagemaker` library includes telemetry to help us understand user needs and improve the SDK.  You can opt-out by setting the `TelemetryOptOut` parameter to `true` in the SDK configuration; see the [SageMaker documentation](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for details.

### AWS Permissions

The SageMaker Python SDK requires the necessary permissions to interact with Amazon SageMaker and other AWS services.  Refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details on required permissions.

### Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License. See [http://aws.amazon.com/apache2.0/](http://aws.amazon.com/apache2.0/) for the full license.

### Running Tests

Install testing dependencies:

```bash
pip install --upgrade .[test]
```

or, for Zsh users:

```bash
pip install --upgrade .\[test]
```

**Unit Tests**

Run unit tests using tox:

```bash
tox tests/unit
```

**Integration Tests**

Integration tests require AWS credentials and a role named `SageMakerRole` with appropriate permissions.

Run integration tests:

```bash
tox -- tests/integ
```

Or, run them in parallel:

```bash
tox -- -n auto tests/integ
```

You can also run specific tests:

```bash
tox -- -k 'test_i_care_about'
```

### Git Hooks

Enable git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

### Building Sphinx Docs

1.  Set up a Python environment and install the dependencies from `doc/requirements.txt`:

```bash
# conda
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

# pip
pip install -r doc/requirements.txt
```

2.  Install the local version of the SDK:

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

## SageMaker SparkML Serving

SageMaker SparkML Serving allows you to perform predictions against SparkML models within SageMaker. Models should be serialized using the MLeap library.

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

For information about the `content-type`, `Accept` formats, and the structure of the `schema`, see the [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container) documentation.
```

Key improvements and explanations:

*   **SEO Optimization:**  Keywords like "SageMaker Python SDK," "Machine Learning," "AWS," "training," "deployment," and framework names are used naturally throughout the text.
*   **Clear and Concise Hook:** The first sentence directly states what the SDK *does* and its key benefit.
*   **Organized Structure:**  Headings, subheadings, and bullet points improve readability and make it easier for users to scan and understand the content.
*   **Table of Contents:**  Provides easy navigation to different sections of the documentation. The links target the sections already available at ReadTheDocs.
*   **Focus on Benefits:** The "Key Features" section highlights the advantages of using the SDK.
*   **Installation and Testing Instructions:** Updated and simplified to include both pip and source installations.  Testing instructions are clarified.
*   **Telemetry & Permissions:**  Added clear explanations of telemetry opt-out and permissions.
*   **SparkML Serving Detail:**  Included the example and link to the SparkML Serving docs to provide the use-case.
*   **Maintained Original Information:** Kept the badges and links from the original README for project credibility.
*   **Clearer Instructions:**  Improved clarity in the "Running Tests" and "Building Sphinx Docs" sections.
*   **Conciseness:** Removed redundant or less critical details, focusing on the core value proposition and essential information.
*   **Corrected Typos/Formatting:** Corrected minor typos and formatting issues.
*   **Direct Links to Documentation:**  Used links to the official ReadTheDocs where possible to improve user access.