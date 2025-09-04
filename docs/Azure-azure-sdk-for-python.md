# Azure SDK for Python: Simplify Cloud Development

**Unlock the power of Microsoft Azure with the official Azure SDK for Python, providing robust and easy-to-use libraries for seamless cloud integration.**  [Explore the Azure SDK for Python on GitHub](https://github.com/Azure/azure-sdk-for-python)

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features:

*   **Comprehensive Service Coverage:** Access a wide range of Azure services with dedicated client libraries.
*   **Simplified Authentication:** Effortlessly authenticate with Azure services using the Azure Identity library.
*   **Consistent Design:** Benefit from a unified design and coding style across all Python libraries.
*   **Robust Functionality:** Enjoy features like retries, logging, and transport protocols built into the core libraries.
*   **Production-Ready Libraries:** Utilize stable, non-preview libraries for production environments.
*   **Management Libraries:** Manage and provision Azure resources through dedicated management libraries.

## Getting Started

The Azure SDK for Python offers a modular approach, with individual libraries for each Azure service. To get started:

1.  **Choose a library:** Select the specific service library you need (e.g., `azure-storage-blob`).  Find available libraries in the `/sdk` directory.
2.  **Explore the documentation:**  Refer to the `README.md` (or `README.rst`) file within the library's project folder for detailed instructions.
3.  **Consult the public developer docs:** For further information, consult our [public developer docs](https://docs.microsoft.com/python/azure/) or our versioned [developer docs](https://azure.github.io/azure-sdk-for-python).

### Prerequisites

The client libraries are supported on Python 3.9 or later. See the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.

## Available Packages

Azure SDK for Python packages are categorized as Client and Management libraries and can be found [here](https://azure.github.io/azure-sdk/releases/latest/index.html#python)

### Client: New Releases

These libraries are **GA** or in **preview**, offering the latest features for interacting with Azure resources. These libraries allow you to use and consume existing resources and interact with them, for example: upload a blob. These libraries share several core functionalities such as: retries, logging, transport protocols, authentication protocols, etc. that can be found in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library. You can learn more about these libraries by reading guidelines that they follow [here](https://azure.github.io/azure-sdk/python/guidelines/index.html).

### Client: Previous Versions

These are the last stable versions of packages for production use, offering similar functionality to new releases with wider service coverage.

### Management: New Releases

New management libraries built according to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) are now available. These offer core capabilities like Azure Identity, HTTP Pipeline with custom policies, error handling, and distributed tracing.
Documentation and code samples for these new libraries can be found [here](https://aka.ms/azsdk/python/mgmt). In addition, a migration guide that shows how to transition from older versions of libraries is located [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

### Management: Previous Versions

For complete list of management libraries that enable you to provision and manage Azure resources, please [check here](https://azure.github.io/azure-sdk/releases/latest/all/python.html).

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** File issues on [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:** Ask questions on StackOverflow using the `azure` and `python` tags.

## Data Collection and Telemetry

This software collects data to improve products and services. You can opt-out of telemetry.

### Telemetry Configuration

Telemetry is enabled by default.  To disable, create a `NoUserAgentPolicy` class and pass an instance during client creation as shown in the example.

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

Report security issues to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

Contributions are welcome!  See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details.