# Azure SDK for Python: Simplify Cloud Development

**Build powerful and reliable applications for the cloud with the Azure SDK for Python, offering comprehensive libraries for accessing and managing Azure services.**  ([Original Repo](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features:

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, networking, databases, and more.
*   **Modular Libraries:** Utilize service-specific libraries for focused development and reduced dependencies.
*   **Up-to-date:** Regularly updated packages to consume and interact with existing resources.
*   **Production Ready:** Stable, non-preview libraries available.
*   **Management Capabilities:** Provision and manage Azure resources effectively with the management libraries.
*   **Azure SDK Design Guidelines:** Uses industry standards for easy use and integration.
*   **Authentication:** Intuitive Azure Identity library simplifies authentication.
*   **Core Capabilities:** Includes features like retries, logging, transport protocols, and authentication protocols found in the azure-core library.
*   **Extensive Documentation and Samples:** Get started quickly with detailed documentation and code samples.
*   **Flexible Versioning:** Access both new releases and previous versions based on your project needs.

## Getting Started

Each Azure service has a dedicated Python library. Explore the `/sdk` directory to find the relevant libraries and their respective `README.md` or `README.rst` files for instructions.

### Prerequisites

The Azure SDK for Python supports Python 3.9 or later.  See the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.

## Packages Available

Libraries are categorized into:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

*   **Description:** Includes **GA** (Generally Available) and preview libraries allowing use and interaction with Azure resources.
*   **Core Functionality:** Shares core features like retries, logging, and authentication, found in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.
*   **Guidelines:** Follows the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/index.html).
*   **Find Packages:** [Latest packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python)
*   **Note:**  For production use, use stable, non-preview libraries.

### Client: Previous Versions

*   **Description:** Stable, production-ready versions of packages.
*   **Functionality:** Provides similar resource access as preview versions.
*   **Considerations:** May not fully implement the latest guidelines or have the same features as the newest releases but offer broader service coverage.

### Management: New Releases

*   **Description:** New management libraries following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/).
*   **Features:** Includes Azure Identity, an HTTP pipeline, error handling, and distributed tracing.
*   **Documentation:** [Documentation and code samples](https://aka.ms/azsdk/python/mgmt).
*   **Migration Guide:** [Migration guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide)
*   **Find Packages:** [Latest management packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)
*   **Note:** If upgrading management libraries, refer to the migration guide for authentication changes.

### Management: Previous Versions

*   **Description:** Libraries for provisioning and managing Azure resources.
*   **Find Packages:** [Complete list](https://azure.github.io/azure-sdk/releases/latest/all/python.html)
*   **Identification:** Identified by namespaces starting with `azure-mgmt-`, e.g., `azure-mgmt-compute`.
*   **Considerations:** May have different feature sets than new releases but offer wider service coverage.

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   [StackOverflow](https://stackoverflow.com/questions/tagged/azure+python) (use tags `azure` and `python`)

## Data Collection

The SDK may collect data to provide services and improve products. Learn more in Microsoft's [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is enabled by default.

To disable it, use the `NoUserAgentPolicy` class (subclass of `UserAgentPolicy`) during client construction.  This disables telemetry for that client.

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

## Reporting Security Issues and Security Bugs

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.  You should receive a response within 24 hours.  More information can be found in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details.  Contributions require a Contributor License Agreement (CLA).

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.