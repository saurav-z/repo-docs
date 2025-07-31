![SageMaker Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models with Ease

**The Amazon SageMaker Python SDK empowers data scientists and developers to build, train, and deploy machine learning models quickly and efficiently on Amazon SageMaker.** Explore the original repository on [GitHub](https://github.com/aws/sagemaker-python-sdk).

## Key Features of the SageMaker Python SDK

*   **Simplified Model Training:** Train models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Built-in Algorithm Support:** Utilize scalable, optimized Amazon algorithms tailored for SageMaker and GPU training.
*   **Bring Your Own Algorithms:**  Train and host models using custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Framework Flexibility:** Seamlessly integrate with frameworks like PyTorch, Scikit-learn, XGBoost, and Chainer.
*   **Model Deployment Options:**  Deploy models with various deployment options.
*   **Automated Model Tuning:**  Leverage SageMaker's Automatic Model Tuning capabilities for optimal performance.
*   **Model Monitoring:** Monitor models for performance and data drift.
*   **Batch Transform:** Process large datasets efficiently with Batch Transform.
*   **Secure Training and Inference:** Securely train and deploy models within a VPC.
*   **SparkML Integration:** Deploy and perform predictions against SparkML models using the MLeap library with SageMaker SparkML Serving.

## Installing the SageMaker Python SDK

Get started by installing the latest version from PyPI:

```bash
pip install sagemaker
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

## Supported Operating Systems and Python Versions

*   **Operating Systems:** Unix/Linux and Mac
*   **Python Versions:**
    *   3.9
    *   3.10
    *   3.11
    *   3.12

## Telemetry

The SDK includes telemetry to help improve functionality. Opt-out by setting the ``TelemetryOptOut`` parameter to ``true`` as described in the documentation.

## AWS Permissions

Amazon SageMaker requires specific permissions to perform operations on your behalf. See the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for detailed information. If using an IAM role with a path, ensure you have permissions for `iam:GetRole`.

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.

## Testing

Run tests using tox or selectively target integration tests:

```bash
tox tests/unit
tox -- -k 'test_i_care_about'  # Filter tests
tox -- tests/integ             # Run integration tests
tox -- -n auto tests/integ     # Run integration tests in parallel
```

## Git Hooks

Enable Git hooks for code quality checks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

## Building Sphinx Docs

Build documentation using:

```bash
conda create -n sagemaker python=3.12  # or preferred version
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0
pip install -r doc/requirements.txt
pip install --upgrade .
cd sagemaker-python-sdk/doc
make html
```

View the website at http://localhost:8000.

## SageMaker SparkML Serving

Deploy SparkML models using the SageMaker SparkML Serving Container:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For details on content types and schemas, see the  [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).