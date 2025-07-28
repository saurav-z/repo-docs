[![SageMaker Python SDK Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)](https://github.com/aws/sagemaker-python-sdk)

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models at Scale

**Simplify your machine learning workflow on Amazon SageMaker with the powerful and versatile SageMaker Python SDK.** This open-source library empowers data scientists and developers to train and deploy machine learning models efficiently.

[View the original repository on GitHub](https://github.com/aws/sagemaker-python-sdk)

**Key Features:**

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow, as well as PyTorch, Chainer, Scikit-learn, and XGBoost.
*   **Amazon Algorithms:** Leverage scalable, optimized implementations of core machine learning algorithms.
*   **Custom Algorithms:** Bring your own algorithms packaged in SageMaker-compatible Docker containers.
*   **Model Deployment:** Easily deploy your trained models for real-time or batch inference.
*   **Model Tuning:** Utilize SageMaker's automatic model tuning capabilities.
*   **Batch Transform:** Perform batch predictions on large datasets.
*   **Secure Training and Inference:** Securely train and deploy models within a VPC.
*   **Integration with other services:** Supports various SageMaker features like Model Monitoring, Debugger and Processing.
*   **SageMaker SparkML Serving:** Perform predictions against a SparkML Model in SageMaker, using the `MLeap` library

**Installation:**

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

**Supported Operating Systems:**

*   Unix/Linux
*   Mac

**Supported Python Versions:**

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

**AWS Permissions:**

The SDK generally requires the same permissions as using SageMaker directly. Refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details.  If you use an IAM role with a path, grant permission for ``iam:GetRole``.

**Telemetry:**

The library has telemetry enabled, which can be opted out of by configuring the ``TelemetryOptOut`` parameter to ``true`` (see the [documentation](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk)).

**Licensing:**

Licensed under the Apache 2.0 License. Copyright Amazon.com, Inc. or its affiliates.
http://aws.amazon.com/apache2.0/

**Testing:**

Run unit tests with:

```bash
tox tests/unit
```

Run integration tests with:

```bash
tox -- tests/integ
```

**SparkML Serving:**

The SageMaker SparkML Serving feature allows you to deploy SparkML models serialized with the MLeap library.

Example deployment:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

Use the predictor to predict with a CSV payload:

```python
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

See [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container) for details.

**Documentation:**

For comprehensive documentation and API reference, visit [Read the Docs](https://sagemaker.readthedocs.io/).