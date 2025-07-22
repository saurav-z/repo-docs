# Azure SDK for Python: Build Powerful Python Applications with Azure

**Leverage the Azure SDK for Python to effortlessly build, deploy, and manage robust applications on the Microsoft Azure cloud platform.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features:

*   **Comprehensive Azure Service Coverage:** Access a wide array of Azure services through dedicated Python libraries, including compute, storage, networking, databases, and more.
*   **Simplified Development:** Benefit from intuitive APIs and consistent design patterns, making it easier to interact with Azure services.
*   **Cross-Platform Compatibility:** Develop and deploy applications on various platforms, including Windows, Linux, and macOS.
*   **Robust Authentication & Security:** Securely connect to Azure using various authentication methods and leverage built-in security features.
*   **Regular Updates:** Stay up-to-date with the latest Azure features and improvements through frequent library updates.
*   **Client and Management Libraries:** Provides access to both client libraries for interacting with Azure resources, and management libraries to provision and manage them.

## Getting Started

Each Azure service has dedicated libraries for seamless integration.  Refer to the `README.md` (or `README.rst`) file within each service's project folder to get started. Service libraries are located in the `/sdk` directory.

### Prerequisites

*   Python 3.9 or later. See the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for more details.

## Available Packages

The Azure SDK for Python offers packages categorized as:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries enable you to consume existing resources and interact with them (e.g., upload a blob) and include new **GA** (Generally Available) and **preview** releases.  They share core functionalities like retries, logging, and authentication, as defined in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library. Review the [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html) for library design. Find the latest packages [here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** For production use, opt for stable, non-preview libraries.

### Client: Previous Versions

These are stable, production-ready versions of packages.  They provide similar functionalities to the preview releases and offer wider service coverage, although they may not follow all recent guidelines.

### Management: New Releases

New management libraries adhering to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) are available, offering core capabilities shared across Azure SDKs, including Azure Identity, HTTP Pipeline, error handling, and distributed tracing. Documentation and code samples are [here](https://aka.ms/azsdk/python/mgmt), and a migration guide is [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).  Find the latest packages [here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> **Note:** Use stable libraries for production. If you experience authentication issues after upgrading management packages, consult the migration guide.

### Management: Previous Versions

For a comprehensive list of management libraries, see [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html).  They provide broad service coverage, though not the same feature set as the new releases. Management libraries have namespaces starting with `azure-mgmt-` (e.g., `azure-mgmt-compute`).

## Need Help?

*   Comprehensive documentation: [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   File an issue: [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   Search or ask questions: StackOverflow using `azure` and `python` tags.

## Data Collection

This software collects usage information, which may be sent to Microsoft. This data helps to provide services, improve products, and is subject to the [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). More data details are available at the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is enabled by default.

To disable telemetry:

1.  Create a `NoUserAgentPolicy` class (subclass of `UserAgentPolicy`) with an empty `on_request` method.
2.  Pass an instance of `NoUserAgentPolicy` as `user_agent_policy` during client creation.

Example (using `azure-storage-blob`):

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

Report security issues to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours. Further information is available in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).