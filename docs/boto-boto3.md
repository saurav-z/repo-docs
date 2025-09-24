# Boto3: The AWS SDK for Python - Simplify AWS Cloud Interactions

Boto3 is the official Amazon Web Services (AWS) SDK for Python, enabling developers to easily integrate AWS services into their Python applications.  [View the original repository on GitHub](https://github.com/boto/boto3).

[![Version](http://img.shields.io/pypi/v/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![Python Versions](https://img.shields.io/pypi/pyversions/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![License](http://img.shields.io/pypi/l/boto3.svg?style=flat)](https://github.com/boto/boto3/blob/develop/LICENSE)

## Key Features

*   **Simplified AWS Integration:** Easily interact with a wide range of AWS services like Amazon S3, Amazon EC2, and many more.
*   **Pythonic API:** Designed to feel natural for Python developers.
*   **Comprehensive Service Support:**  Access to a vast array of AWS services.  Check the  `doc site <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>`_ for the most up-to-date list of supported services.
*   **Active Development and Support:** Maintained and published by Amazon Web Services, ensuring ongoing updates and support.
*   **Credential Management:** Securely manage your AWS credentials.

## Getting Started

### Prerequisites

*   A supported version of Python.

### Installation

1.  **Set up a virtual environment** (recommended):

    ```bash
    $ python -m venv .venv
    $ . .venv/bin/activate
    ```

2.  **Install using pip:**

    ```bash
    $ python -m pip install boto3
    ```

    Alternatively, to install from source:

    ```bash
    $ git clone https://github.com/boto/boto3.git
    $ cd boto3
    $ python -m pip install -r requirements.txt
    $ python -m pip install -e .
    ```

### Configuration

1.  **Configure AWS Credentials:**  Set up your AWS credentials in either:
    *   `~/.aws/credentials`:

        ```ini
        [default]
        aws_access_key_id = YOUR_KEY
        aws_secret_access_key = YOUR_SECRET
        ```
    *   or use environment variables.
    *   or using IAM roles (best practice for EC2 instances).

2.  **Set Default Region (optional but recommended):** Configure your default region in:
    *   `~/.aws/config`:

        ```ini
        [default]
        region=us-east-1
        ```

    See `here <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html>`_ for more configuration methods.

### Example Usage

```python
import boto3
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)
```

## Running Tests

To run the tests, use `tox`:

```bash
$ tox
```

You can also run individual tests with pytest. See the original README for more detailed instructions.

## Getting Help

*   **Stack Overflow:** Ask questions and tag them with `boto3 <https://stackoverflow.com/questions/tagged/boto3>`.
*   **AWS Support:** Open a support ticket with `AWS Support <https://console.aws.amazon.com/support/home#>`.
*   **GitHub Issues:** Report bugs or suggest features by  `opening an issue <https://github.com/boto/boto3/issues/new>`.

## Contributing

We welcome community contributions! Please review the `CONTRIBUTING <https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst>`_ document before submitting any issues or pull requests.

## Maintenance and Support

Boto3 was released on 06/22/2015 and is currently in the full support phase. For information about the SDK's maintenance policy, refer to:

*   `AWS SDKs and Tools Maintenance Policy <https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html>`_
*   `AWS SDKs and Tools Version Support Matrix <https://docs.aws.amazon.com/sdkref/latest/guide/version-support-matrix.html>`_

## More Resources

*   `NOTICE <https://github.com/boto/boto3/blob/develop/NOTICE>`_
*   `Changelog <https://github.com/boto/boto3/blob/develop/CHANGELOG.rst>`_
*   `License <https://github.com/boto/boto3/blob/develop/LICENSE>`_