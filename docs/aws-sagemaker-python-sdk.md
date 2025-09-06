[![SageMaker Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)](https://github.com/aws/sagemaker-python-sdk)

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models

**The Amazon SageMaker Python SDK empowers you to build, train, and deploy machine learning models seamlessly on Amazon SageMaker.**

*   **Comprehensive Framework Support:** Train models using popular deep learning frameworks like Apache MXNet and TensorFlow, as well as PyTorch, Chainer, Scikit-learn, and XGBoost.
*   **Built-in Algorithms:** Leverage scalable, optimized Amazon algorithms for core machine learning tasks.
*   **Bring Your Own Algorithms:** Train and deploy models using custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Simplified Deployment:** Easily deploy your trained models with just a few lines of code.
*   **Integration with MLeap:** Deploy and perform predictions against a SparkML Model in SageMaker using MLeap serialization.

[View the original repository on GitHub](https://github.com/aws/sagemaker-python-sdk)

## Key Features

*   **Model Training:** Train models using various frameworks and algorithms.
*   **Model Deployment:** Deploy trained models to production environments.
*   **Framework Support:** Extensive support for popular ML frameworks.
*   **Built-in Algorithms:** Access to pre-built, optimized algorithms.
*   **Custom Algorithm Support:** Utilize your own Docker containers for training and hosting.
*   **Model Monitoring:** Comprehensive model monitoring capabilities for tracking performance and drift.
*   **Model Debugging:** Advanced debugging tools to identify and resolve model issues.
*   **Batch Transform:** Efficiently process large datasets for inference.
*   **SparkML Serving:** Deploy and perform predictions against a SparkML Model in SageMaker using MLeap serialization.

## Getting Started

### Installing the SageMaker Python SDK

Install the latest version using `pip`:

```bash
pip install sagemaker
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Python Versions

The SDK is tested on:

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

## Resources

*   **Documentation:** [Read the Docs](https://sagemaker.readthedocs.io/en/stable/)
*   **API Reference:** [Read the Docs](https://sagemaker.readthedocs.io/)

## Additional Information

*   **AWS Permissions:** For required permissions, see the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).  If using an IAM role with a path, grant permission for `iam:GetRole`.
*   **Telemetry:**  The library includes telemetry to understand usage. Opt-out by setting `TelemetryOptOut` to `true` in the SDK defaults configuration.  See [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for details.
*   **Licensing:**  Licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).
*   **Running Tests:**  Run unit and integration tests using `tox`.  Install test dependencies with `pip install --upgrade .[test]`.

### SageMaker SparkML Serving

SageMaker SparkML Serving allows you to perform predictions against a SparkML Model serialized with the MLeap library.

*   **Supported Spark Version:** 3.3 (MLeap version - 0.20.0)

Example:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more on `content-type`, `Accept` formats, and the `schema`, see [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).