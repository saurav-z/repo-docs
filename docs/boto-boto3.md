# Boto3: The AWS SDK for Python - Simplify Cloud Interactions

Boto3 is the powerful and versatile AWS SDK for Python, empowering developers to seamlessly integrate with Amazon Web Services.  Explore the official [Boto3 repository on GitHub](https://github.com/boto/boto3) to get started.

[![Version](http://img.shields.io/pypi/v/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![Python Versions](https://img.shields.io/pypi/pyversions/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![License](http://img.shields.io/pypi/l/boto3.svg?style=flat)](https://github.com/boto/boto3/blob/develop/LICENSE)

## Key Features of Boto3:

*   **Simplified AWS Service Integration:** Interact with services like Amazon S3, Amazon EC2, and many more using Python code.
*   **Comprehensive Documentation:** Access the latest and most up-to-date documentation via the [Boto3 Documentation Site](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html).
*   **Pythonic API:** Leverage a Python-friendly interface for easy development and reduced boilerplate code.
*   **Maintained by AWS:** Benefit from regular updates, bug fixes, and new feature releases from Amazon Web Services.

## Getting Started

### Prerequisites

Ensure you have a supported version of Python installed.

### Installation

1.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv .venv
    . .venv/bin/activate
    ```

2.  **Install Boto3:**

    ```bash
    python -m pip install boto3
    ```

    or install from source:
    ```bash
    git clone https://github.com/boto/boto3.git
    cd boto3
    python -m pip install -r requirements.txt
    python -m pip install -e .
    ```

### Configuration

1.  **Set up AWS Credentials:** Configure your AWS credentials using one of the following methods:

    *   **`~/.aws/credentials` (Recommended):**

        ```ini
        [default]
        aws_access_key_id = YOUR_KEY
        aws_secret_access_key = YOUR_SECRET
        ```

    *   **`~/.aws/config` (for default region):**

        ```ini
        [default]
        region=us-east-1
        ```

    *   Refer to the [AWS documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html) for other configuration methods.

### Example Usage

```python
import boto3

# Create an S3 resource
s3 = boto3.resource('s3')

# Iterate through buckets
for bucket in s3.buckets.all():
    print(bucket.name)
```

## Running Tests

Use `tox` to run tests in all supported Python versions:

```bash
tox
```

Run specific tests:

```bash
tox -- unit/test_session.py
tox -e py26,py33 -- integration/
```

Run individual tests with your default Python version:

```bash
pytest tests/unit
```

## Getting Help

*   **GitHub Issues:** Report bugs and request features on [GitHub Issues](https://github.com/boto/boto3/issues).
*   **Stack Overflow:** Ask questions on [Stack Overflow](https://stackoverflow.com/) and tag them with `boto3`.
*   **AWS Support:** Open a support ticket with [AWS Support](https://console.aws.amazon.com/support/home#/).

## Contributing

We welcome contributions! Please review the [CONTRIBUTING](https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst) document before submitting pull requests.

## Maintenance and Support

*   See the [AWS SDKs and Tools Maintenance Policy](https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html) and [AWS SDKs and Tools Version Support Matrix](https://docs.aws.amazon.com/sdkref/latest/guide/version-support-matrix.html) for information on SDK major versions and their dependencies.

## More Resources

*   [NOTICE](https://github.com/boto/boto3/blob/develop/NOTICE)
*   [Changelog](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)
*   [License](https://github.com/boto/boto3/blob/develop/LICENSE)