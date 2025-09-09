# Boto3: The AWS SDK for Python - Simplify Your Cloud Interactions

Boto3 is the official AWS SDK for Python, empowering developers to build robust applications that seamlessly integrate with Amazon Web Services.  Explore the [original Boto3 repository](https://github.com/boto/boto3) for more details.

[![Version](http://img.shields.io/pypi/v/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![Python Versions](https://img.shields.io/pypi/pyversions/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![License](http://img.shields.io/pypi/l/boto3.svg?style=flat)](https://github.com/boto/boto3/blob/develop/LICENSE)

## Key Features

*   **Comprehensive AWS Service Support:** Interact with a wide range of AWS services like S3, EC2, DynamoDB, and many more.
*   **Simplified Development:** Write Python code to manage and access AWS resources with ease.
*   **Authentication and Authorization:** Securely access your AWS resources using various authentication methods.
*   **Resource-Oriented Programming:** Utilize high-level resource objects for simplified interactions with services.
*   **Active Community and Support:** Benefit from a large and active community, and AWS support resources.

## Getting Started

### Prerequisites

*   A supported version of Python installed.

### Installation

1.  **Set up a virtual environment** (recommended):

    ```bash
    python -m venv .venv
    . .venv/bin/activate
    ```

2.  **Install Boto3 using pip:**

    ```bash
    python -m pip install boto3
    ```

    Or, to install from source:
    ```bash
    git clone https://github.com/boto/boto3.git
    cd boto3
    python -m pip install -r requirements.txt
    python -m pip install -e .
    ```

### Configuration

1.  **Configure AWS Credentials:**

    Set up your AWS credentials (access key ID and secret access key) in your local `~/.aws/credentials` file:

    ```ini
    [default]
    aws_access_key_id = YOUR_KEY
    aws_secret_access_key = YOUR_SECRET
    ```
2.  **Set up Default Region:**

    Configure the default AWS region (e.g., us-east-1) in `~/.aws/config`:

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
Or run specific tests, e.g.:

```bash
tox -- unit/test_session.py
```

## Getting Help

*   **GitHub Issues:**  Report bugs and request features on [GitHub](https://github.com/boto/boto3/issues/new).
*   **Stack Overflow:**  Ask questions and get answers on [Stack Overflow](https://stackoverflow.com/) with the `boto3` tag.
*   **AWS Support:** Open a support ticket with [AWS Support](https://console.aws.amazon.com/support/home#/).

## Contributing

Contributions are welcome! Please review the [CONTRIBUTING](https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst) document before submitting any issues or pull requests.

## Maintenance and Support

Boto3 was made generally available on 06/22/2015 and is currently in the full support phase of the availability life cycle.  See the [AWS SDKs and Tools Maintenance Policy](https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html) and [AWS SDKs and Tools Version Support Matrix](https://docs.aws.amazon.com/sdkref/latest/guide/version-support-matrix.html) for more information.

## More Resources

*   [NOTICE](https://github.com/boto/boto3/blob/develop/NOTICE)
*   [Changelog](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)
*   [License](https://github.com/boto/boto3/blob/develop/LICENSE)
*   [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)