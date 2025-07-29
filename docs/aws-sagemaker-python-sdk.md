[![SageMaker Python SDK Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)](https://github.com/aws/sagemaker-python-sdk)

# SageMaker Python SDK: Train and Deploy Machine Learning Models on AWS

**The SageMaker Python SDK simplifies the process of building, training, and deploying machine learning models on Amazon SageMaker, offering flexibility and ease of use for developers.**

*   **Seamless Model Training and Deployment:** Train and deploy models using popular deep learning frameworks like Apache MXNet, TensorFlow, PyTorch, and others, including custom algorithms within Docker containers.
*   **Built-in Algorithm Support:** Utilize scalable Amazon algorithms optimized for SageMaker and GPU training, including XGBoost and more.
*   **Flexibility and Customization:** Train and host models with your own algorithms built into SageMaker-compatible Docker containers.
*   **Integration with Key Services:** Leverage features like automatic model tuning, batch transform, model monitoring, and more for comprehensive model lifecycle management.
*   **Comprehensive Documentation:** Access detailed documentation, including an API reference, via [Read the Docs](https://sagemaker.readthedocs.io).

[View the original repository on GitHub](https://github.com/aws/sagemaker-python-sdk)

## Key Features

*   **Framework Support:**  Extensive support for popular frameworks like TensorFlow, PyTorch, MXNet, and scikit-learn.
*   **Algorithm Integration:** Easy access to built-in Amazon algorithms.
*   **Custom Algorithm Support:** Train and deploy models using custom algorithms packaged in Docker containers.
*   **Model Deployment:** Simplify model deployment with flexible deployment options.
*   **Integration with SageMaker Features:** Leverage features like automatic model tuning, batch transform, and model monitoring.
*   **SparkML Serving:** Deploy and perform predictions against SparkML models serialized with MLeap.

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

### Telemetry

The SDK has telemetry enabled to help understand user needs. You can opt out by setting the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration. More information is available [here](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk).

### AWS Permissions

As a managed service, Amazon SageMaker needs permissions to operate on your behalf.  Review the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for necessary permissions.

If using an IAM role with a path, grant permission for `iam:GetRole`.

##  SageMaker SparkML Serving

With SageMaker SparkML Serving, you can deploy SparkML models serialized with MLeap.  This allows you to perform predictions against your SparkML model in SageMaker.

### Example Deployment

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

###  Invoking the Endpoint

You can invoke the deployed endpoint with a CSV payload:

```python
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more details on content-type and accept formats, refer to the [SageMaker SparkML Serving Container documentation](https://github.com/aws/sagemaker-sparkml-serving-container).

## Licensing

The SageMaker Python SDK is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).

## Testing

The SDK provides unit and integration tests.

### Test Prerequisites

*   Install testing dependencies with `pip install --upgrade .[test]` or `pip install --upgrade .\[test\]` for Zsh.
*   **Unit Tests:** Run unit tests with tox: `tox tests/unit`. Ensure interpreters for supported Python versions are installed.
*   **Integration Tests:** Require AWS account credentials and an IAM role named `SageMakerRole` with the `AmazonSageMakerFullAccess` policy, plus the necessary permissions for Elastic Inference.
*   Create a dummy ECR repo for remote function tests.

### Running Tests

*   Run all integration tests sequentially: `tox -- tests/integ`. (May take a while)
*   Run integration tests in parallel: `tox -- -n auto tests/integ`.
*   Filter integration tests by function name: `tox -- -k 'test_i_care_about'`.

## Git Hooks

Enable git hooks in the `.git/hooks` directory:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

To enable an individual hook, move it from the `.githooks/` to the `.git/hooks/` directory.

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

2.  Clone/fork the repo and install:

    ```bash
    pip install --upgrade .
    ```

3.  Build the docs:

    ```bash
    cd sagemaker-python-sdk/doc
    make html
    ```

4.  Preview the site:

    ```bash
    cd _build/html
    python -m http.server 8000
    ```

    View the website at `http://localhost:8000`.