[![SageMaker Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)](https://github.com/aws/sagemaker-python-sdk)

# SageMaker Python SDK: Train and Deploy Machine Learning Models on AWS

**Supercharge your machine learning workflow with the SageMaker Python SDK, an open-source library for seamless model training and deployment on Amazon SageMaker.**  ([View the GitHub Repository](https://github.com/aws/sagemaker-python-sdk))

## Key Features:

*   **Flexible Framework Support:** Train models using popular deep learning frameworks like Apache MXNet, TensorFlow, PyTorch, and more.
*   **Built-in Algorithms:** Leverage Amazon's scalable, optimized machine learning algorithms for efficient training on SageMaker and GPU instances.
*   **Bring Your Own Algorithms:**  Easily train and host models using your custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Model Deployment:** Deploy your trained models with just a few lines of code.
*   **Model Serving:** Host SparkML models in SageMaker and perform predictions.
*   **Comprehensive Documentation:** Access detailed documentation, including an API reference, at [Read the Docs](https://sagemaker.readthedocs.io).
*   **Supports the latest Python versions:** 3.9, 3.10, 3.11, and 3.12

## Getting Started

### Installing the SageMaker Python SDK

Install the latest version using pip:

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

*   Unix/Linux
*   Mac

## Telemetry

The `sagemaker` library has telemetry enabled to help us better understand user needs. If you prefer to opt out, set `TelemetryOptOut` to `true` in the SDK defaults configuration.  See [Configuring and using defaults](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for more information.

## AWS Permissions

Amazon SageMaker requires certain permissions to perform operations on your behalf. Review the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details.

## Testing

Run tests to validate the code.

Install testing libraries:
```bash
pip install --upgrade .[test]
```

Run unit tests:
```bash
tox tests/unit
```

Run integration tests:
```bash
tox -- tests/integ
```

## SageMaker SparkML Serving

Perform predictions against a SparkML model in SageMaker using the `SparkMLModel` class.

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

## Licensing

The SageMaker Python SDK is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).

## Building Sphinx Docs

1.  Set up a Python environment and install dependencies from `doc/requirements.txt`.
2.  Clone/fork the repo and install the local version.
3.  Navigate to the `doc` directory and run `make html`.
4.  View the site by running a Python web server and visiting http://localhost:8000.