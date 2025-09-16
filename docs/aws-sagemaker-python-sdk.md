<div align="center">
<img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">
</div>

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models

**Unlock the power of Amazon SageMaker with the official Python SDK, simplifying the entire machine learning lifecycle.**  This library provides a Pythonic interface to train, deploy, and manage your machine learning models on Amazon SageMaker.  Explore the official repository on [GitHub](https://github.com/aws/sagemaker-python-sdk).

## Key Features

*   **Flexible Training:** Train models using popular deep learning frameworks like Apache MXNet, TensorFlow, PyTorch, and others, or utilize Amazon's built-in algorithms.  Supports custom algorithms packaged in Docker containers.
*   **Simplified Deployment:** Easily deploy your trained models for real-time or batch inference.
*   **Model Management:** Manage model versions, monitor performance, and automate the machine learning workflow.
*   **Integration with Amazon Services:** Seamlessly integrates with other AWS services like S3, IAM, and CloudWatch.
*   **Extensive Documentation:** Comprehensive documentation, including API references and tutorials, available on [Read the Docs](https://sagemaker.readthedocs.io/en/stable/).

## Key Features

*   Train models using popular deep learning frameworks such as Apache MXNet, TensorFlow, PyTorch, and others.
*   Train and deploy models with Amazon algorithms, optimized for SageMaker and GPU training.
*   Support for training and hosting models with custom algorithms built into SageMaker compatible Docker containers.
*   Simplified deployment for real-time or batch inference.
*   Seamlessly integrates with other AWS services.
*   Includes model monitoring and automated machine learning.

## Installation

Install the latest version from PyPI using pip:

```bash
pip install sagemaker
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

## Usage

The SageMaker Python SDK provides a high-level API for common machine learning tasks.  Here are some basic examples.  For more in-depth guides and examples, refer to the [official documentation](https://sagemaker.readthedocs.io/en/stable/).

*   **Training a Model:**
    ```python
    from sagemaker.tensorflow import TensorFlow

    # Configure the TensorFlow estimator
    tf_estimator = TensorFlow(
        entry_point='train.py', # Replace with your training script
        role='arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE',
        instance_count=1,
        instance_type='ml.m5.large',
        framework_version='2.15', # Check the SageMaker documentation for latest supported versions
        py_version='py310'
    )

    # Train the model
    tf_estimator.fit({'training': 's3://YOUR_BUCKET/YOUR_TRAINING_DATA/'})
    ```

*   **Deploying a Model:**
    ```python
    predictor = tf_estimator.deploy(initial_instance_count=1, instance_type='ml.m5.large')
    ```

*   **Making Predictions:**
    ```python
    import json

    payload = json.dumps({'instances': [{'feature1': 1.0, 'feature2': 2.0}]})
    predictions = predictor.predict(payload)
    print(predictions)
    ```

## Key Features and Functionality

*   **Training:**
    *   Supports training with popular frameworks (TensorFlow, PyTorch, MXNet, etc.)
    *   Use of SageMaker's built-in algorithms.
    *   Bring Your Own Container (BYOC) support for custom algorithms.

*   **Deployment:**
    *   Deploy models for real-time inference.
    *   Batch transform for large-scale inference.

*   **Additional Features:**
    *   Automatic Model Tuning (Hyperparameter optimization).
    *   Model Monitoring for performance analysis.
    *   Integration with other AWS services.

## Supported Operating Systems

*   Unix/Linux
*   Mac

## Supported Python Versions

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

## Telemetry

The ``sagemaker`` library has telemetry enabled to help us better understand user needs, diagnose issues, and deliver new features.  You can opt-out by setting the ``TelemetryOptOut`` parameter to ``true`` in the SDK defaults configuration. More info available at [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk).

## AWS Permissions

The SDK requires permissions for SageMaker operations. Read more about required permissions in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html). If using an IAM role with a path, grant permission for ``iam:GetRole``.

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License. See http://aws.amazon.com/apache2.0/ for details.

## Testing

To run tests, install testing dependencies:

```bash
pip install --upgrade .[test]
```

Run unit tests using tox:

```bash
tox tests/unit
```

Run integration tests:

```bash
tox -- tests/integ
```

## Git Hooks

Enable git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

## Building Documentation

Follow these steps to build the documentation:

1.  Set up a Python environment and install dependencies:

    ```bash
    # conda
    conda create -n sagemaker python=3.12
    conda activate sagemaker
    conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

    # pip
    pip install -r doc/requirements.txt
    ```

2.  Install the project:

    ```bash
    pip install --upgrade .
    ```

3.  Build the HTML documentation:

    ```bash
    cd doc
    make html
    ```

4.  View the documentation (optional):

    ```bash
    cd _build/html
    python -m http.server 8000
    ```

    Then, visit http://localhost:8000.

## SageMaker SparkML Serving

Leverage the SageMaker Python SDK to perform predictions on SparkML models using the MLeap library.

### Prerequisites

*   SparkML models serialized using MLeap.
*   Supported Spark version: 3.3 (with MLeap version 0.20.0).

### Usage Example

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

### Making Predictions

Use a CSV payload for prediction:

```python
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more details on accepted content types, formats, and the schema structure, see the [SageMaker SparkML Serving Container documentation](https://github.com/aws/sagemaker-sparkml-serving-container).