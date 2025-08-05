[![SageMaker Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)](https://github.com/aws/sagemaker-python-sdk)

# SageMaker Python SDK: Train and Deploy Machine Learning Models on AWS

**The SageMaker Python SDK empowers data scientists and developers to easily build, train, and deploy machine learning models on Amazon SageMaker.** ([View Original Repo](https://github.com/aws/sagemaker-python-sdk))

## Key Features

*   **Flexible Training:** Supports training models with popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Built-in Algorithms:** Utilize scalable, optimized Amazon algorithms for various ML tasks.
*   **Bring Your Own Algorithms:** Train and deploy models using your custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Model Deployment:** Deploy trained models with ease using a variety of instance types and configurations.
*   **Framework Support:** Full support for major frameworks like MXNet, TensorFlow, PyTorch, Chainer, Scikit-learn, and XGBoost.
*   **Integration with ML Platforms:** Includes support for Amazon SageMaker Operators in Apache Airflow, and SageMaker SparkML Serving.
*   **Advanced Features:** Includes integrations for SageMaker Autopilot, Model Monitoring, and Debugger.

## Getting Started

### Installing the SageMaker Python SDK

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

### Supported Operating Systems

*   Unix/Linux
*   Mac

### Supported Python Versions

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

### Telemetry

The SDK has telemetry enabled. You can opt-out by setting the `TelemetryOptOut` parameter to `true`.  See the [SageMaker documentation](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for details.

### AWS Permissions

The SageMaker Python SDK requires the necessary permissions for using SageMaker. Refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for information on required permissions.

If using an IAM role with a path, grant permission for `iam:GetRole`.

## Additional Information

### Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.

### Running Tests

Install testing dependencies:

```bash
pip install --upgrade .[test]
```

Run unit tests using tox:

```bash
tox tests/unit
```

Run integration tests (requires AWS credentials and a configured IAM role):

```bash
tox tests/integ
```

### Git Hooks

To enable all git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

### Building Sphinx Documentation

1.  Set up a Python environment and install dependencies:

    ```bash
    # conda
    conda create -n sagemaker python=3.12
    conda activate sagemaker
    conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

    # pip
    pip install -r doc/requirements.txt
    ```
2.  Install your local version:

    ```bash
    pip install --upgrade .
    ```
3.  Build the documentation:

    ```bash
    cd sagemaker-python-sdk/doc
    make html
    ```

4.  Preview the site:

    ```bash
    cd _build/html
    python -m http.server 8000
    ```

    Then visit `http://localhost:8000`.

## SageMaker SparkML Serving

Deploy and perform predictions against a SparkML Model in SageMaker, serialized with the MLeap library.

Supported major version of Spark: 3.3 (MLeap version - 0.20.0)

```python
from sagemaker.sparkml.model import SparkMLModel

sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more information, see the `SageMaker SparkML Serving Container`_.

.. _SageMaker SparkML Serving Container: https://github.com/aws/sagemaker-sparkml-serving-container