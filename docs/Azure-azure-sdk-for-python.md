# Azure SDK for Python: Simplify Cloud Development with Python

**The Azure SDK for Python empowers developers to seamlessly build, deploy, and manage applications on Microsoft Azure.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Azure Service Coverage:** Access a wide range of Azure services, including compute, storage, networking, databases, and more.
*   **Intuitive Pythonic Design:**  Leverages the power of Python with an intuitive API, making it easy to integrate Azure services into your Python applications.
*   **Client and Management Libraries:** Access services with client libraries for interacting with resources, and management libraries to provision and manage resources.
*   **Robust Authentication:** Built-in support for various authentication methods, including Azure Active Directory and managed identities, for secure access to Azure resources.
*   **Cross-Platform Compatibility:** Supports Python 3.9 and later, enabling development on Windows, macOS, and Linux.
*   **Up-to-Date Packages:** Stay current with the newest features with new releases and packages.

## Getting Started

Get started with a specific service library by exploring the `README.md` (or `README.rst`) file located in the library's project folder within the `/sdk` directory.

### Prerequisites

The client libraries are supported on Python 3.9 or later.  For more details, please read our [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

Choose from a selection of libraries categorized by function:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These GA and Preview packages allow you to use existing Azure resources and interact with them (e.g., upload a blob). These libraries share common functionalities found in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library. Learn more about the guidelines [here](https://azure.github.io/azure-sdk/python/guidelines/index.html).

Find the [latest packages here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Important:** Use stable, non-preview libraries for production code.

### Client: Previous Versions

Use these stable, production-ready packages for access to Azure services.

### Management: New Releases

Leveraging [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), these libraries offer core capabilities like Azure Identity, HTTP pipeline policies, and more.

*   Documentation and code samples [here](https://aka.ms/azsdk/python/mgmt).
*   Migration guide [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

Find the [latest packages here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> **Important:** Ensure you use stable libraries for production use and review the migration guide if upgrading packages.

### Management: Previous Versions

For a complete list of management libraries [check here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). These libraries are identified by namespaces starting with `azure-mgmt-`, such as `azure-mgmt-compute`.

## Need Help?

*   **Documentation:**  Visit our [Azure SDK for Python documentation](https://aka.ms/python-docs).
*   **Issues:** File an issue via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues).
*   **Community:** Check [previous questions](https://stackoverflow.com/questions/tagged/azure+python) or ask new ones on StackOverflow using `azure` and `python` tags.

## Data Collection

The software collects information about your use of the software and sends it to Microsoft. You can learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is enabled by default.  To disable it, pass a `NoUserAgentPolicy` instance during client creation.

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

### Reporting Security Issues

Report security issues to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details.  This project uses the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).