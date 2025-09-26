# Boto3: The AWS SDK for Python

**Simplify your interaction with Amazon Web Services using Boto3, the official Python SDK.**

[View the original repository on GitHub](https://github.com/boto/boto3)

Boto3 is the Amazon Web Services (AWS) Software Development Kit (SDK) for Python, empowering developers to build robust applications that leverage AWS services like Amazon S3, Amazon EC2, and many more. This SDK provides a Pythonic interface to interact with AWS, simplifying complex tasks and enabling efficient cloud management.

## Key Features:

*   **Comprehensive AWS Service Coverage:** Supports a wide array of AWS services.
*   **Easy-to-Use Interface:** Provides a Python-friendly API for interacting with AWS.
*   **Credential Management:** Seamlessly integrates with various credential configuration methods.
*   **Asynchronous Operations:** Supports asynchronous operations for improved performance.
*   **Actively Maintained by AWS:** Benefit from regular updates, bug fixes, and new service integrations.

## Getting Started

### Prerequisites:

*   Python (See `Python Versions` badge at the top for supported versions.)

### Installation:

1.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    . .venv/bin/activate
    ```

2.  **Install Boto3 using pip:**

    ```bash
    python -m pip install boto3
    ```

### Configuration:

1.  **Configure AWS Credentials:**
    Set up your AWS credentials, typically in `~/.aws/credentials`:

    ```ini
    [default]
    aws_access_key_id = YOUR_KEY
    aws_secret_access_key = YOUR_SECRET
    ```
2.  **Set a Default Region:**
    Configure your default AWS region, usually in `~/.aws/config`:

    ```ini
    [default]
    region=us-east-1
    ```

    More credential configuration options can be found [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

### Example Usage:

```python
import boto3
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)
```

## Running Tests

Boto3 utilizes `tox` for running tests across multiple Python versions.

1.  **Run all tests:**

    ```bash
    tox
    ```
2.  **Run specific tests:**

    ```bash
    tox -- unit/test_session.py
    ```
3.  **Run tests with a specific Python version:**

    ```bash
    tox -e py310
    ```

## Help and Support

*   **Documentation:** [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
*   **Community Support:**
    *   Ask questions on [Stack Overflow](https://stackoverflow.com/) with the tag `boto3`.
    *   Contact [AWS Support](https://console.aws.amazon.com/support/home#/).
*   **Report Issues:**  [Open an issue on GitHub](https://github.com/boto/boto3/issues/new) for bugs or feature requests.

## Contributing

We welcome community contributions!  Please review the [CONTRIBUTING](https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst) guidelines before submitting pull requests.

## Maintenance and Support

For details on the maintenance policy and version support, see:

*   [AWS SDKs and Tools Maintenance Policy](https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html)
*   [AWS SDKs and Tools Version Support Matrix](https://docs.aws.amazon.com/sdkref/latest/guide/version-support-matrix.html)

## Additional Resources

*   [NOTICE](https://github.com/boto/boto3/blob/develop/NOTICE)
*   [Changelog](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)
*   [License](https://github.com/boto/boto3/blob/develop/LICENSE)