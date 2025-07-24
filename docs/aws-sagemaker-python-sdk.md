<div align="center">
  <a href="https://github.com/aws/sagemaker-python-sdk">
    <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">
  </a>
  <h1>SageMaker Python SDK: Simplify Machine Learning on AWS</h1>
</div>

The SageMaker Python SDK is your gateway to effortlessly train, tune, and deploy machine learning models on Amazon SageMaker.

**Key Features:**

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Built-in Algorithms:** Leverage scalable, optimized Amazon algorithms for core machine learning tasks.
*   **Bring Your Own Algorithms:** Seamlessly integrate your custom algorithms through SageMaker-compatible Docker containers.
*   **Model Deployment:** Easily deploy trained models with a few lines of code.
*   **Model Packaging:** Effortlessly package your models for use on SageMaker.
*   **Model Tuning:** Simplify the process of optimizing your model hyperparameters with automation.
*   **Model Monitoring:** Gain visibility into your models' performance and detect anomalies.
*   **Model Debugging:** Effectively debug and analyze model performance.
*   **Batch Transform:** Apply your model to process large datasets efficiently.
*   **SparkML Serving:** Easily perform predictions against SparkML Models within SageMaker.

**Getting Started**

*   **[Original Repository](https://github.com/aws/sagemaker-python-sdk)**: Access the source code and contribute.
*   **Documentation:** Access detailed documentation, including the API reference at `Read the Docs <https://sagemaker.readthedocs.io>`_.

**Installation**

Install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

**Supported Versions**

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

**SageMaker SparkML Serving**

Deploy and perform predictions against SparkML models on SageMaker, serialized with the `MLeap` library.

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

**Additional Resources**

*   **AWS Permissions:**  Consult the `AWS Documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`_ for required permissions.
*   **Licensing:** This SDK is licensed under the Apache 2.0 License.