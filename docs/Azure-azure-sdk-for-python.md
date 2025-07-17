# Azure SDK for Python: Build Cloud Applications with Ease

**Easily build and manage applications on Microsoft Azure with the official Azure SDK for Python.**

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

The Azure SDK for Python provides a comprehensive set of libraries to interact with Azure services, enabling developers to build robust, scalable, and secure cloud applications.  Find the source code on [GitHub](https://github.com/Azure/azure-sdk-for-python).

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services including compute, storage, networking, databases, and more.
*   **Modern Design:** Benefit from libraries that follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), ensuring consistency and ease of use.
*   **Production-Ready:** Choose from stable, non-preview libraries for reliable performance in your production environments.
*   **Enhanced Capabilities**: Benefit from several core capabilities such as: retries, logging, transport protocols, authentication protocols, etc.
*   **Up-to-date Packages:** Find the [most up to date list of all of the new packages on our page](https://azure.github.io/azure-sdk/releases/latest/index.html#python)

## Getting Started

The Azure SDK for Python uses service-specific libraries.

*   **Prerequisites:**  Requires Python 3.9 or later. See the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.
*   **Service Libraries:** Explore the `/sdk` directory to locate service-specific libraries. Refer to each library's `README.md` or `README.rst` file for detailed instructions.

## Package Categories

Explore available packages, categorized by release status and service type:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are GA (Generally Available) and preview packages to consume existing resources and interact with them. These libraries have several core functionalities such as: retries, logging, transport protocols, authentication protocols, etc.

### Client: Previous Versions

Last stable versions of packages for Azure and are production-ready. These libraries provide similar functionalities to the Preview ones.

### Management: New Releases

Management libraries follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/).

### Management: Previous Versions

Libraries to provision and manage Azure resources.

## Need Help?

*   **Documentation:** Visit the [Azure SDK for Python documentation](https://aka.ms/python-docs) for detailed information.
*   **Issues:** Report issues via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues).
*   **Community:**  Search [previous questions](https://stackoverflow.com/questions/tagged/azure+python) or ask on StackOverflow using `azure` and `python` tags.

## Data Collection & Telemetry

The software collects information about your use, which is sent to Microsoft to improve products and services. You can learn more about data collection in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) or on the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry collection is on by default. To opt out, you can disable telemetry at client construction.

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

Report security issues and bugs privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details.  All contributions require a Contributor License Agreement (CLA).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).