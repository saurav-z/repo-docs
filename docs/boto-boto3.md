# Boto3: The AWS SDK for Python - Simplify AWS Cloud Development

**Boto3** is the official Amazon Web Services (AWS) SDK for Python, empowering developers to seamlessly integrate and manage AWS services like S3, EC2, and more. [View the source on GitHub](https://github.com/boto/boto3).

[![Version](http://img.shields.io/pypi/v/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![Python Versions](https://img.shields.io/pypi/pyversions/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![License](http://img.shields.io/pypi/l/boto3.svg?style=flat)](https://github.com/boto/boto3/blob/develop/LICENSE)

## Key Features

*   **Comprehensive AWS Service Support:** Access a vast array of AWS services directly from your Python applications.
*   **Simplified AWS Integration:** Interact with AWS resources using intuitive and Pythonic APIs.
*   **Easy Credential Management:** Supports various authentication methods, including environment variables, configuration files, and IAM roles.
*   **Well-Documented:** Benefit from extensive documentation and examples to quickly learn and implement AWS services.
*   **Maintained by AWS:** Receive ongoing support, updates, and new feature releases from Amazon Web Services.

## Getting Started

### Prerequisites

*   A supported version of Python installed (check the shields above).

### Installation

1.  **Set up a virtual environment:**

    ```bash
    python -m venv .venv
    . .venv/bin/activate
    ```

2.  **Install Boto3 using pip:**

    ```bash
    python -m pip install boto3
    ```

    Alternatively, you can install from source:

    ```bash
    git clone https://github.com/boto/boto3.git
    cd boto3
    python -m pip install -r requirements.txt
    python -m pip install -e .
    ```

### Configuration

1.  **Configure your AWS credentials:**  (e.g., in `~/.aws/credentials`)

    ```ini
    [default]
    aws_access_key_id = YOUR_KEY
    aws_secret_access_key = YOUR_SECRET
    ```

2.  **Set a default region:** (e.g., in `~/.aws/config`)

    ```ini
    [default]
    region=us-east-1
    ```

    More credential configuration methods can be found [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

### Example Usage

```python
import boto3
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)
```

## Running Tests

You can run tests using `tox`:

```bash
tox
```

Run individual tests or specify `pytest` options:

```bash
tox -- unit/test_session.py
tox -e py26,py33 -- integration/
pytest tests/unit
```

## Getting Help

*   Ask a question on [Stack Overflow](https://stackoverflow.com/) and tag it with `boto3`.
*   Open a support ticket with [AWS Support](https://console.aws.amazon.com/support/home#/).
*   Report potential bugs by [opening an issue](https://github.com/boto/boto3/issues/new).

## Contributing

We welcome your contributions! Please review the [CONTRIBUTING](https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst) document before submitting any issues or pull requests.

## Maintenance and Support

Boto3 was made generally available on 06/22/2015 and is currently in the full support phase.
For information about maintenance and support for SDK major versions, see the following in the AWS SDKs and Tools Shared Configuration and Credentials Reference Guide:

*   [AWS SDKs and Tools Maintenance Policy](https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html)
*   [AWS SDKs and Tools Version Support Matrix](https://docs.aws.amazon.com/sdkref/latest/guide/version-support-matrix.html)

## More Resources

*   [NOTICE](https://github.com/boto/boto3/blob/develop/NOTICE)
*   [Changelog](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)
*   [License](https://github.com/boto/boto3/blob/develop/LICENSE)