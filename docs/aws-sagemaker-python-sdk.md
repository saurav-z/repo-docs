![SageMaker Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models

**Accelerate your machine learning journey with the Amazon SageMaker Python SDK, a powerful open-source library for seamless training and deployment on Amazon SageMaker.**  [View the original repository on GitHub](https://github.com/aws/sagemaker-python-sdk).

**Key Features:**

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet, TensorFlow, PyTorch, Chainer, and Scikit-learn.
*   **Amazon Algorithms:** Utilize scalable, optimized implementations of core machine learning algorithms.
*   **Bring Your Own Algorithms:** Train and host models using your custom algorithms packaged in SageMaker compatible Docker containers.
*   **Model Deployment:** Easily deploy trained models for real-time or batch inference.
*   **Integration with MLeap:**  Serve SparkML models using MLeap library.
*   **Comprehensive Documentation:** Access detailed documentation, including API references, on [Read the Docs](https://sagemaker.readthedocs.io/en/stable/).
*   **Support for Multiple Python Versions:** Tested on Python 3.9, 3.10, 3.11, and 3.12.

**Get Started:**

**Installation:**

```bash
pip install sagemaker  # Installs the latest version from PyPI
```

or, for development from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

**Supported Operating Systems:**

*   Unix/Linux
*   Mac

**Usage Examples:**

```python
from sagemaker.tensorflow.estimator import TensorFlow
# Example: Deploy a TensorFlow model.
tf_estimator = TensorFlow(entry_point='train.py',
                        role='arn:aws:iam::123456789012:role/SageMakerRole',
                        instance_count=1,
                        instance_type='ml.m5.large',
                        framework_version='2.15')

tf_estimator.fit({'training': 's3://your-bucket/training-data'})
predictor = tf_estimator.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge')
```

```python
from sagemaker.sparkml.model import SparkMLModel

# Example: Deploy a SparkML model.
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name='sparkml-endpoint')
```

**Additional Information:**

*   **AWS Permissions:** Learn about the necessary permissions for using Amazon SageMaker in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).
*   **Telemetry:** The SDK has telemetry enabled to help us understand user needs. Opt-out by setting `TelemetryOptOut` to `true`.
*   **Licensing:**  The SageMaker Python SDK is licensed under the Apache 2.0 License.
*   **Testing:**  Run unit and integration tests using `tox`. See the original `README` for detailed instructions on running tests.
*   **Building Documentation:** Detailed instructions on building the Sphinx documentation are available in the `README`.
*   **SageMaker SparkML Serving:** Perform predictions against SparkML models deployed in SageMaker.  Models must be serialized with the MLeap library.