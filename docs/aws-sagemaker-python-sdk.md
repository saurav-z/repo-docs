[![SageMaker](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)](https://github.com/aws/sagemaker-python-sdk)

# SageMaker Python SDK: Train and Deploy Machine Learning Models at Scale

**The SageMaker Python SDK empowers data scientists and developers to easily build, train, and deploy machine learning models on Amazon SageMaker.**  This SDK simplifies the entire ML lifecycle, from data preparation to model deployment and monitoring.  [Explore the original repository](https://github.com/aws/sagemaker-python-sdk).

## Key Features:

*   **Framework Support:** Seamlessly train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Built-in Algorithms:** Utilize scalable, optimized Amazon algorithms for core machine learning tasks.
*   **Custom Algorithms:** Bring your own algorithms packaged in SageMaker-compatible Docker containers.
*   **Simplified Deployment:**  Easily deploy models to SageMaker endpoints for real-time or batch inference.
*   **Model Monitoring:** Monitor model performance and drift to ensure accuracy and reliability.
*   **Integration with ML Tools:** Utilize SageMaker with tools like Apache Airflow, Debugger, and Processing.
*   **SparkML Serving:** Deploy SparkML models serialized with the MLeap library.

## Getting Started

### Installation

Install the latest version using pip:

```bash
pip install sagemaker
```

Or install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Platforms

*   **Operating Systems:** Unix/Linux and Mac.
*   **Python Versions:** 3.9, 3.10, 3.11, and 3.12.

### Essential AWS Permissions

The SDK leverages Amazon SageMaker, a managed service, to perform operations on your behalf on the AWS hardware that is managed by Amazon SageMaker. SageMaker can perform only operations that the user permits. For more details, refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html). Additionally, if you're using an IAM role with a path, grant permission for ``iam:GetRole``.

### Telemetry

The `sagemaker` library has telemetry enabled to help us better understand user needs, diagnose issues, and deliver new features. To opt-out, set the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration. For more information, visit the [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk).

## Examples & Resources

### Using SparkML Serving

Deploy and perform predictions against your SparkML models using MLeap.

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

### Additional Resources

*   **Documentation:**  [Read the Docs](https://sagemaker.readthedocs.io/)
*   **SageMaker SparkML Serving Container:** [Link](https://github.com/aws/sagemaker-sparkml-serving-container)

## Contributing

We welcome contributions! See the [CONTRIBUTING](https://github.com/aws/sagemaker-python-sdk/blob/master/CONTRIBUTING.md) guide for more information.

## License

The SageMaker Python SDK is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).