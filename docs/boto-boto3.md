# Boto3: The AWS SDK for Python - Simplify Your Cloud Interactions

Boto3 is the official Amazon Web Services (AWS) SDK for Python, empowering developers to easily integrate and manage AWS services directly from their Python applications.  [View the original repository](https://github.com/boto/boto3).

[![Version](http://img.shields.io/pypi/v/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![Python Versions](https://img.shields.io/pypi/pyversions/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![License](http://img.shields.io/pypi/l/boto3.svg?style=flat)](https://github.com/boto/boto3/blob/develop/LICENSE)

## Key Features

*   **Comprehensive AWS Service Support:** Interact with a wide range of AWS services, including Amazon S3, Amazon EC2, DynamoDB, and many more.
*   **Simplified API:**  Provides a Pythonic and intuitive interface for interacting with AWS services.
*   **Authentication and Authorization:**  Handles AWS credentials securely, simplifying authentication and authorization.
*   **Resource-Oriented Programming:** Leverages a resource-oriented programming model for easier interaction with AWS resources.
*   **Up-to-Date Documentation:**  Access the latest documentation and supported services directly on the [Boto3 documentation site](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html).

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

3.  **Alternatively, install from source:**

    ```bash
    git clone https://github.com/boto/boto3.git
    cd boto3
    python -m pip install -r requirements.txt
    python -m pip install -e .
    ```

### Configuration

1.  **Set up AWS credentials:**
    Configure your AWS credentials by either setting environment variables or adding them to the `~/.aws/credentials` file:

    ```ini
    [default]
    aws_access_key_id = YOUR_KEY
    aws_secret_access_key = YOUR_SECRET
    ```

2.  **Set a default region:**
    Configure a default region in the `~/.aws/config` file:

    ```ini
    [default]
    region=us-east-1
    ```

    More credential configuration methods are available in the [Boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

### Example Usage

```python
import boto3
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)
```

## Running Tests

Run tests across all supported Python versions using `tox`:

```bash
tox
```

Run tests with specific pytest options:

```bash
tox -- unit/test_session.py
tox -e py26,py33 -- integration/
```

Or run individual tests with your default Python version:

```bash
pytest tests/unit
```

## Getting Help

*   Ask questions on [Stack Overflow](https://stackoverflow.com/) and tag them with `boto3`.
*   Open a support ticket with [AWS Support](https://console.aws.amazon.com/support/home#/).
*   Report bugs by [opening an issue](https://github.com/boto/boto3/issues/new) on GitHub.

## Contributing

We welcome contributions!  Review the [CONTRIBUTING](https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst) document before submitting issues or pull requests.

## Maintenance and Support for SDK Major Versions

Boto3 was made generally available on 06/22/2015 and is currently in the full support phase.

For information about maintenance and support for SDK major versions and their underlying dependencies, see the following in the AWS SDKs and Tools Shared Configuration and Credentials Reference Guide:

*   [AWS SDKs and Tools Maintenance Policy](https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html)
*   [AWS SDKs and Tools Version Support Matrix](https://docs.aws.amazon.com/sdkref/latest/guide/version-support-matrix.html)

## More Resources

*   [NOTICE](https://github.com/boto/boto3/blob/develop/NOTICE)
*   [Changelog](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)
*   [License](https://github.com/boto/boto3/blob/develop/LICENSE)