# Boto3: The AWS SDK for Python - Simplify AWS Integration

Boto3 is the official AWS SDK for Python, empowering developers to seamlessly integrate their applications with a wide array of AWS services.  **[View the original repository on GitHub](https://github.com/boto/boto3)**.

[![Version](http://img.shields.io/pypi/v/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![Python Versions](https://img.shields.io/pypi/pyversions/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![License](http://img.shields.io/pypi/l/boto3.svg?style=flat)](https://github.com/boto/boto3/blob/develop/LICENSE)

## Key Features of Boto3:

*   **Simplified AWS Integration:** Easily interact with AWS services like S3, EC2, DynamoDB, and many more.
*   **Pythonic Interface:**  Provides a Python-friendly API for interacting with AWS resources.
*   **Comprehensive Service Coverage:** Supports a vast and growing number of AWS services.
*   **Up-to-Date Documentation:**  Access the latest documentation on the official [Boto3 documentation site](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html).
*   **Actively Maintained:** Maintained and published by Amazon Web Services.

## Getting Started

### Prerequisites
*   Ensure you have a supported version of Python installed.

### Installation

1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    . .venv/bin/activate
    ```

2.  **Install Boto3 using pip:**
    ```bash
    python -m pip install boto3
    ```

3.  **(Optional) Install from source:**
    ```bash
    git clone https://github.com/boto/boto3.git
    cd boto3
    python -m pip install -r requirements.txt
    python -m pip install -e .
    ```

## Using Boto3

1.  **Configure your AWS credentials:**

    *   Set your AWS credentials in the `~/.aws/credentials` file:
        ```ini
        [default]
        aws_access_key_id = YOUR_KEY
        aws_secret_access_key = YOUR_SECRET
        ```
    *   Configure the default region in `~/.aws/config`:
        ```ini
        [default]
        region=us-east-1
        ```
    *   For other credential configuration methods, refer to the [AWS documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

2.  **Example Usage:**

    ```python
    import boto3
    s3 = boto3.resource('s3')
    for bucket in s3.buckets.all():
        print(bucket.name)
    ```

## Running Tests

You can run tests for Boto3 using `tox`:

```bash
tox
```

To run specific tests:

```bash
tox -- unit/test_session.py
tox -e py26,py33 -- integration/
pytest tests/unit
```

## Getting Help

*   Ask a question on [Stack Overflow](https://stackoverflow.com/) and tag it with `boto3`.
*   Open a support ticket with [AWS Support](https://console.aws.amazon.com/support/home#/).
*   Report a bug by [opening an issue](https://github.com/boto/boto3/issues/new) on GitHub.

## Contributing

We welcome community contributions!  Please review the [CONTRIBUTING document](https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst) before submitting any issues or pull requests.

## Maintenance and Support

Boto3 was made generally available on 06/22/2015 and is currently in the full support phase of the availability life cycle.

*   [AWS SDKs and Tools Maintenance Policy](https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html)
*   [AWS SDKs and Tools Version Support Matrix](https://docs.aws.amazon.com/sdkref/latest/guide/version-support-matrix.html)

## Additional Resources

*   [NOTICE](https://github.com/boto/boto3/blob/develop/NOTICE)
*   [Changelog](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)
*   [License](https://github.com/boto/boto3/blob/develop/LICENSE)