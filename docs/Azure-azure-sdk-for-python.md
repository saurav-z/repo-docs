# Azure SDK for Python: Build Powerful Python Applications on Azure

**Unlock the full potential of Microsoft Azure with the official Azure SDK for Python, providing a comprehensive suite of libraries for building and managing cloud applications.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Library Coverage:** Access a wide range of Azure services with dedicated Python libraries, including storage, compute, databases, AI, and more.
*   **Modern Design Guidelines:** Benefit from consistent and intuitive APIs that adhere to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), ensuring a smooth developer experience.
*   **Robust Core Functionality:** Leverage shared features like retries, logging, authentication, and transport protocols provided by the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.
*   **Management and Client Libraries:** Access management libraries for provisioning and managing Azure resources and client libraries to use and consume existing resources.
*   **Cross-Platform Compatibility:** Built for Python 3.9 or later, the SDK integrates seamlessly with various operating systems and environments.
*   **Detailed Documentation and Examples:** Find extensive documentation and code samples to help you get started quickly and effectively.

## Getting Started

The Azure SDK for Python offers modular libraries for individual Azure services, allowing you to choose only the dependencies you need. To get started, find the `README.md` (or `README.rst`) file within each service library's project folder in the `/sdk` directory.

### Prerequisites

Ensure you have Python 3.9 or later installed. For detailed version support information, refer to the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

Explore a range of libraries categorized as follows:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

New, GA (General Availability) and preview client libraries empower you to interact with existing Azure resources. These libraries incorporate core functionalities like retries, logging, and authentication found in the `azure-core` library.

*   Find the latest package releases on our [releases page](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

    > **Note:** For production use, we recommend stable, non-preview libraries.

### Client: Previous Versions

Stable versions of client libraries are production-ready for interacting with Azure services, offering functionality similar to preview versions.

### Management: New Releases

New management libraries align with the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), providing shared capabilities such as Azure Identity, an HTTP Pipeline, error handling, and distributed tracing.

*   Documentation and code samples are available [here](https://aka.ms/azsdk/python/mgmt).
*   A migration guide is available [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).
*   Find the latest management package releases on our [releases page](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

    > **Note:** For production use, use stable libraries. Refer to the migration guide for authentication issues after upgrading.

### Management: Previous Versions

Management libraries provide a comprehensive set of tools for provisioning and managing Azure resources.

*   Find a complete list of libraries [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html).
*   Management libraries use namespaces starting with `azure-mgmt-`.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Stack Overflow:** Search for questions with the `azure` and `python` tags.

## Data Collection

The software collects telemetry data to improve services. You can opt-out of telemetry using the methods described below. For more details, see the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) and Microsoft's [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704).

### Telemetry Configuration

Telemetry is enabled by default. To disable telemetry, define a `NoUserAgentPolicy` class and pass it as the `user_agent_policy` during client construction.

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

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

Contributions are welcome! See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md).