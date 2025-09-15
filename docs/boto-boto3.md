# Boto3: The AWS SDK for Python

**Boto3 empowers Python developers to effortlessly interact with Amazon Web Services, simplifying cloud application development.**  [Learn more about Boto3 on GitHub](https://github.com/boto/boto3).

[![PyPI Version](http://img.shields.io/pypi/v/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![Python Versions](https://img.shields.io/pypi/pyversions/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![License](http://img.shields.io/pypi/l/boto3.svg?style=flat)](https://github.com/boto/boto3/blob/develop/LICENSE)

## Key Features

*   **Simplified AWS Integration:** Easily interact with services like Amazon S3, Amazon EC2, and many more.
*   **Pythonic API:** Designed to feel natural and intuitive for Python developers.
*   **Comprehensive Documentation:** Access up-to-date documentation for all supported AWS services.
*   **Actively Maintained:**  Developed and maintained by Amazon Web Services.

## Getting Started

### Prerequisites

*   A supported version of Python installed.

### Installation

1.  **Set up a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    . .venv/bin/activate
    ```

2.  **Install Boto3 using pip:**

    ```bash
    python -m pip install boto3
    ```

    Alternatively, install from source:

    ```bash
    git clone https://github.com/boto/boto3.git
    cd boto3
    python -m pip install -r requirements.txt
    python -m pip install -e .
    ```

### Configuration

1.  **Set up your AWS credentials (e.g., in `~/.aws/credentials`):**

    ```ini
    [default]
    aws_access_key_id = YOUR_KEY
    aws_secret_access_key = YOUR_SECRET
    ```

2.  **Set up a default region (e.g., in `~/.aws/config`):**

    ```ini
    [default]
    region=us-east-1
    ```

    More credential configuration methods are available [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

### Example Usage

```python
import boto3
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)
```

## Running Tests

Run tests using `tox`:

```bash
tox
```

For more specific test runs:

```bash
tox -- unit/test_session.py
tox -e py26,py33 -- integration/
pytest tests/unit
```

## Getting Help

*   Ask questions on [Stack Overflow](https://stackoverflow.com/) and tag them with `boto3`.
*   Open a support ticket with [AWS Support](https://console.aws.amazon.com/support/home#/).
*   Report bugs by [opening an issue](https://github.com/boto/boto3/issues/new) on GitHub.

## Contributing

Contributions are welcome!  Please review the [CONTRIBUTING](https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst) document before submitting pull requests or issues.

## Maintenance and Support

Boto3 was made generally available on 06/22/2015 and is currently in the full support phase of the availability life cycle.

For information about maintenance and support for SDK major versions and their underlying dependencies, see the following in the AWS SDKs and Tools Shared Configuration and Credentials Reference Guide:

*   [AWS SDKs and Tools Maintenance Policy](https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html)
*   [AWS SDKs and Tools Version Support Matrix](https://docs.aws.amazon.com/sdkref/latest/guide/version-support-matrix.html)

## Additional Resources

*   [NOTICE](https://github.com/boto/boto3/blob/develop/NOTICE)
*   [Changelog](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)
*   [License](https://github.com/boto/boto3/blob/develop/LICENSE)