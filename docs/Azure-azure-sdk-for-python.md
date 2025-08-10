# Azure SDK for Python

**Build powerful Python applications with the official Azure SDK, providing seamless integration with Microsoft Azure cloud services.**

[Link to original repo](https://github.com/Azure/azure-sdk-for-python)

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Coverage:** Access a wide range of Azure services, including compute, storage, networking, databases, and more.
*   **Ease of Use:** Simple and intuitive APIs designed for Python developers, making it easy to integrate Azure services into your applications.
*   **Reliability and Performance:** Built with a focus on performance and reliability, ensuring your applications run smoothly in the cloud.
*   **Authentication and Security:** Securely authenticate to Azure using various methods, including Azure Active Directory and shared access signatures.
*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux, allowing you to develop and deploy your applications on any platform.
*   **Regular Updates:** Benefit from ongoing updates and improvements, ensuring you always have access to the latest features and security patches.

## Getting Started

The Azure SDK for Python offers a modular approach, with individual libraries for each Azure service. This allows you to include only the dependencies you need.

### Prerequisites

*   Python 3.9 or later.

### Find Service Libraries

You can find service libraries in the `/sdk` directory.  Refer to each library's `README.md` (or `README.rst`) file in each project folder for usage instructions.

## Available Packages

The Azure SDK for Python offers packages in the following categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the latest **GA** (Generally Available) and **preview** packages that enable you to consume existing resources and interact with Azure services (e.g., uploading a blob).

*   Sharing core functionalities like retries, logging, transport protocols, and authentication.
*   Follows [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/index.html).

Find the [latest packages here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).  **Note:** Use stable, non-preview libraries for production.

### Client: Previous Versions

These are the last stable versions of packages for production use. They provide similar functionality to the new releases.

### Management: New Releases

New management libraries, following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), are now available.

*   Include capabilities like Azure Identity, HTTP Pipeline, error handling, and distributed tracing.
*   Documentation and code samples: [https://aka.ms/azsdk/python/mgmt](https://aka.ms/azsdk/python/mgmt)
*   Migration guide: [https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide)

Find the [latest management packages here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html). **Note:** If you are experiencing authentication issues after upgrading, refer to the migration guide.

### Management: Previous Versions

Enables you to provision and manage Azure resources; identified by namespaces that start with `azure-mgmt-`, e.g., `azure-mgmt-compute`.

*   View the complete list [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html).

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:** Ask questions on StackOverflow using the `azure` and `python` tags.

## Data Collection

The software collects information about your usage, which is sent to Microsoft to provide services and improve products. You can turn off telemetry. Read the [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry collection is on by default.  To disable it, override the `UserAgentPolicy`:

```python
import os
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient
from azure.core.pipeline.policies import UserAgentPolicy


# Create your credential you want to use
mi_credential = ManagedIdentityCredential()

account_url = "https://<storageaccountname>.blob.core.windows.net"

# Set up user-agent override
class NoUserAgentPolicy(UserAgentPolicy):
    def on_request(self, request):
        pass

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient(account_url, credential=mi_credential, user_agent_policy=NoUserAgentPolicy())

container_client = blob_service_client.get_container_client(container=<container_name>) 
# TODO: do something with the container client like download blob to a file
```

## Reporting Security Issues

Report security issues and bugs privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours.  See the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue) for more information.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details. Contributions require a Contributor License Agreement (CLA).

*   **Code of Conduct:** [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
*   **Contact:** [opencode@microsoft.com](mailto:opencode@microsoft.com)