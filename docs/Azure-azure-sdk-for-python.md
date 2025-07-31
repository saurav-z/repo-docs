# Azure SDK for Python: Simplify Cloud Development

**Easily build and manage applications on Microsoft Azure with the official Azure SDK for Python, offering a comprehensive set of libraries for seamless integration.** [Learn more at the original repo](https://github.com/Azure/azure-sdk-for-python).

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

The Azure SDK for Python provides a rich set of libraries to interact with Azure services. This README provides an overview, key features, and resources for getting started. For in-depth information, refer to the [public developer docs](https://docs.microsoft.com/python/azure/) and the versioned [developer docs](https://azure.github.io/azure-sdk-for-python).

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services through dedicated client libraries.
*   **Modular Design:** Utilize individual libraries for specific Azure services, promoting flexibility and efficiency.
*   **Consistent Experience:** Benefit from shared core functionalities like retries, logging, authentication, and transport protocols, provided by the `azure-core` library.
*   **Management Capabilities:** Leverage management libraries to provision and manage Azure resources.
*   **Adherence to Guidelines:** New management libraries follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/).
*   **Version Support:** Client libraries are supported on Python 3.9 or later.

## Getting Started

### Prerequisites

*   Python 3.9 or later

### Package Structure

Service-specific libraries are located in the `/sdk` directory. To begin, explore the `README.md` (or `README.rst`) file within each library's project folder.

## Packages Available

*   **Client Libraries:**
    *   **New Releases (GA & Preview):** Interact with existing resources (e.g., upload a blob). For a complete list, visit: [Azure SDK for Python Releases](https://azure.github.io/azure-sdk/releases/latest/index.html#python).
    *   **Previous Versions:** Stable, production-ready versions offering similar functionalities.
*   **Management Libraries:**
    *   **New Releases:** Follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). Find documentation and samples [here](https://aka.ms/azsdk/python/mgmt) and a migration guide [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide). For a complete list, visit: [Azure SDK for Python Management Releases](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).
    *   **Previous Versions:** Enable you to provision and manage Azure resources.
        Find a complete list of management libraries [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Management libraries are identified by `azure-mgmt-` namespaces.

> **Important:**  For production environments, it's recommended to use stable, non-preview libraries.

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   Stack Overflow: Use the tags `azure` and `python` to find answers or ask new questions.

## Data Collection

The SDK collects telemetry data.  You can learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704).
For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is enabled by default.  To disable it, use the `NoUserAgentPolicy` class:

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

Report security issues to the Microsoft Security Response Center (MSRC) via <secure@microsoft.com>.

## Contributing

*   See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details.
*   All contributions require a Contributor License Agreement (CLA).
*   This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).