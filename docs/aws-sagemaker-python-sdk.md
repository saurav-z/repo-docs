# SageMaker Python SDK: Train and Deploy Machine Learning Models on AWS

**Unlock the power of Amazon SageMaker with the open-source SageMaker Python SDK, enabling seamless training and deployment of your machine learning models.**  [Explore the original repository](https://github.com/aws/sagemaker-python-sdk).

## Key Features

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like **Apache MXNet**, **TensorFlow**, **PyTorch**, and more.
*   **Amazon Algorithms:** Leverage scalable, optimized implementations of core machine learning algorithms.
*   **Bring Your Own Algorithms (BYOA):** Train and host models with your custom algorithms packaged in SageMaker-compatible Docker containers.
*   **SparkML Serving:** Deploy and perform predictions against SparkML models serialized with MLeap.
*   **Simplified Workflow:** Streamline model training, tuning, and deployment with an easy-to-use Python interface.
*   **Integration with SageMaker Features:** Leverage SageMaker's advanced capabilities, including model monitoring, debugging, automatic model tuning, batch transform, and more.

## Core Functionality

The SageMaker Python SDK provides a comprehensive toolkit for managing your machine learning lifecycle on Amazon SageMaker. Here's what you can do:

*   **Training:** Easily train models using various frameworks, built-in algorithms, or custom containers.
*   **Deployment:** Deploy your trained models to SageMaker endpoints for real-time inference.
*   **Model Tuning:** Optimize model hyperparameters with automatic model tuning.
*   **Batch Transform:** Perform batch predictions on large datasets.
*   **Monitoring:** Track model performance and identify potential issues.

## Installation

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

## Supported Technologies

*   **Supported Python Versions:** 3.9, 3.10, 3.11, and 3.12
*   **Operating Systems:** Unix/Linux and Mac
*   **Frameworks:** Apache MXNet, TensorFlow, Chainer, PyTorch, Scikit-learn, XGBoost, and SparkML.

## Additional Information

*   **Documentation:**  Find detailed documentation, including the API reference, at [Read the Docs](https://sagemaker.readthedocs.io/en/stable/).
*   **Permissions:** The SageMaker Python SDK requires the necessary AWS permissions for accessing SageMaker resources.  Review the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details.
*   **Licensing:**  This SDK is licensed under the Apache 2.0 License.

## Development

*   **Running Tests:**  Use `tox tests/unit` for unit tests and `tox tests/integ` for integration tests.
*   **Building Documentation:** Install dependencies from `doc/requirements.txt`, and then run `make html` in the `doc` directory.

## SageMaker SparkML Serving

Deploy and perform predictions against a SparkML Model in SageMaker. The model should be serialized with MLeap library. For more information on MLeap, see https://github.com/combust/mleap .

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

```