# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models with Ease

**Unlock the power of Amazon SageMaker with the flexible and easy-to-use Python SDK, streamlining your machine learning workflow.** Learn more and contribute at the [SageMaker Python SDK GitHub repository](https://github.com/aws/sagemaker-python-sdk).

## Key Features:

*   **Simplified Model Training:** Train models using popular deep learning frameworks like Apache MXNet, TensorFlow, PyTorch, and more.
*   **Built-in Algorithm Support:** Leverage scalable, optimized Amazon algorithms for core machine learning tasks.
*   **Bring Your Own Algorithms:** Seamlessly integrate your custom algorithms packaged in SageMaker compatible Docker containers.
*   **Model Deployment:** Deploy your trained models with ease for real-time inference.
*   **Comprehensive Documentation:** Access detailed documentation, including API references, via Read the Docs.
*   **Flexible Framework Support:** Works with Chainer, Scikit-learn, XGBoost and SparkML.
*   **Advanced Functionality:** Utilize features like Automatic Model Tuning, Batch Transform, Model Monitoring, and Debugger.

## Installation

Install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker
```

Alternatively, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

## Supported Features

*   **Training and Deployment:** Simplifies the process of training and deploying machine learning models on Amazon SageMaker.
*   **Framework Compatibility:** Supports popular deep learning frameworks like Apache MXNet, TensorFlow, PyTorch, and others.
*   **Amazon Algorithms:** Provides scalable implementations of core machine learning algorithms optimized for SageMaker.
*   **Custom Algorithms:** Enables the use of custom algorithms built into SageMaker compatible Docker containers.
*   **SageMaker SparkML Serving:** Provides the functionality to perform predictions against a SparkML Model in SageMaker using the ``MLeap`` library.

## Additional Information

*   **Supported Operating Systems:** Unix/Linux and Mac.
*   **Supported Python Versions:** 3.9, 3.10, 3.11, and 3.12.
*   **Telemetry:** Telemetry is enabled to help improve the SDK. Opt-out by setting "TelemetryOptOut" to "true" in SDK defaults.
*   **AWS Permissions:** Requires permissions outlined in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).
*   **Licensing:** Licensed under the Apache 2.0 License.

## Running Tests

Install testing dependencies:

```bash
pip install --upgrade .[test]
```

*   **Unit Tests:**  Run unit tests using tox: `tox tests/unit`
*   **Integration Tests:**  Run integration tests with tox, ensuring AWS credentials and a SageMakerRole are configured.

## Building Documentation

Build the documentation using Sphinx:

```bash
# conda (Recommended)
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

# pip
pip install -r doc/requirements.txt
pip install --upgrade .
cd doc
make html
```
View the documentation at `_build/html/`.

## SageMaker SparkML Serving

Perform predictions against a SparkML model in SageMaker:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

## Git Hooks

Enable git hooks for code style and validation:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```