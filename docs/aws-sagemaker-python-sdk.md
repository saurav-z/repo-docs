<!-- Banner Image - Consider using a local file or CDN for faster loading -->
![SageMaker](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)

# SageMaker Python SDK: Train and Deploy Machine Learning Models with Ease

**Accelerate your machine learning journey** with the SageMaker Python SDK, a powerful and flexible library for building, training, and deploying machine learning models on Amazon SageMaker.  [Explore the original repository](https://github.com/aws/sagemaker-python-sdk).

## Key Features:

*   **Simplified Model Training:** Train models using popular deep learning frameworks like Apache MXNet and TensorFlow, as well as pre-built Amazon algorithms and custom algorithms within SageMaker compatible Docker containers.
*   **Flexible Deployment Options:** Deploy trained models with ease, including options for real-time inference and batch transformation.
*   **Support for Various ML Frameworks:**  Seamlessly integrate with frameworks like MXNet, TensorFlow, PyTorch, Chainer, Scikit-learn, and XGBoost.
*   **Model Tuning and Optimization:** Leverage SageMaker's automatic model tuning for optimal performance and efficiency.
*   **Model Monitoring:** Integrate model monitoring to track model performance, identify issues, and maintain model accuracy.
*   **Batch Transform:**  Efficiently process large datasets with batch transform capabilities.
*   **SparkML Serving:** Easily perform predictions against SparkML models in SageMaker using MLeap library.
*   **Built-in Examples & Documentation:** Get started quickly with comprehensive documentation and examples.

## Getting Started

### Installation

Install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker
```

Or, to install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Versions

*   **Python:** 3.9, 3.10, 3.11, 3.12
*   **Operating Systems:** Unix/Linux and Mac

### Telemetry

The SDK has telemetry enabled to collect usage data to help improve the product. You can opt-out by setting the `TelemetryOptOut` parameter to `true` in the SDK configuration.

### AWS Permissions

The SDK requires standard SageMaker permissions, described in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html). If using an IAM role with a path, you may need to grant permission for `iam:GetRole`.

### Licensing

The SageMaker Python SDK is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).

### Testing

Install testing dependencies:

```bash
pip install --upgrade .[test]
```

Run unit tests:

```bash
tox tests/unit
```

Run integration tests (requires AWS credentials and a SageMakerRole IAM role):

```bash
tox -- tests/integ
```

### Building Documentation

1.  **Setup:** Create a Python environment and install dependencies:

    ```bash
    conda create -n sagemaker python=3.12
    conda activate sagemaker
    conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0
    pip install -r doc/requirements.txt
    ```

2.  **Install local version:**

    ```bash
    pip install --upgrade .
    ```

3.  **Build documentation:**

    ```bash
    cd sagemaker-python-sdk/doc
    make html
    ```

4.  **Preview (optional):**

    ```bash
    cd _build/html
    python -m http.server 8000
    ```

    Then, view the site at http://localhost:8000

## SageMaker SparkML Serving

With SageMaker SparkML Serving, you can now perform predictions against a SparkML Model in SageMaker. In order to host a SparkML model in SageMaker, it should be serialized with the ``MLeap`` library.

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

See `SageMaker SparkML Serving Container`_ for more information.