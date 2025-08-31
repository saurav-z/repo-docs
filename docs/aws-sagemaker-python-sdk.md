<div align="center">
  <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">
</div>

# SageMaker Python SDK: Train and Deploy Machine Learning Models

**Easily train and deploy your machine learning models on Amazon SageMaker with the flexible and powerful SageMaker Python SDK.**  [Explore the original repository](https://github.com/aws/sagemaker-python-sdk)

## Key Features

*   **Framework Support:** Train models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Built-in Algorithms:** Leverage scalable Amazon algorithms optimized for SageMaker and GPU training.
*   **Custom Algorithms:**  Train and deploy models using your own algorithms built into SageMaker compatible Docker containers.
*   **Model Deployment:** Simplify model deployment with easy-to-use APIs.
*   **SparkML Serving:** Deploy and perform predictions against SparkML models serialized with the MLeap library.
*   **Comprehensive Documentation:** Detailed documentation and API reference available at [Read the Docs](https://sagemaker.readthedocs.io/en/stable/).
*   **Integration with Common Tools:**  Works seamlessly with tools like Apache Airflow and supports model monitoring and debugging.
*   **Automatic Model Tuning:** Optimize your model performance with SageMaker's automatic model tuning capabilities.

## Core Functionality

The SageMaker Python SDK simplifies the process of building, training, and deploying machine learning models on Amazon SageMaker. It provides a high-level interface for interacting with SageMaker services, allowing you to:

*   **Train Models:**  Utilize pre-built algorithms or bring your own custom training code.
*   **Deploy Models:**  Easily deploy your trained models to production environments.
*   **Manage Resources:**  Create, manage, and monitor SageMaker resources such as training jobs, endpoints, and models.

## Installation

```bash
pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>
```

or install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

## Supported Environments

*   **Operating Systems:** Unix/Linux, Mac
*   **Python Versions:**  3.9, 3.10, 3.11, 3.12

## Telemetry

The SDK collects telemetry data to improve the user experience. You can opt-out by setting the `TelemetryOptOut` parameter to `true`.  See [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for instructions.

## AWS Permissions

Ensure your IAM role has the necessary permissions as documented in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html). Consider granting permission for `iam:GetRole` if you're using an IAM role with a path.

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.  See [http://aws.amazon.com/apache2.0/](http://aws.amazon.com/apache2.0/) for details.

## Testing

*   **Unit Tests:** Run unit tests with tox:  `tox tests/unit`
*   **Integration Tests:**  Run integration tests:  `tox -- tests/integ` (or use filtering options)

## Documentation

Complete documentation, including API references and usage examples, is available on [Read the Docs](https://sagemaker.readthedocs.io/en/stable/).

## Example: Deploying a SparkML Model

Deploy a SparkML model serialized with MLeap:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name='sparkml-endpoint')
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more detailed information, please see [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).