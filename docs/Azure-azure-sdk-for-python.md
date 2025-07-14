# Azure SDK for Python: Develop Powerful Python Applications with Azure (SEO Optimized)

**Empower your Python projects with the official Azure SDK, offering a comprehensive suite of libraries to seamlessly integrate with Microsoft Azure services.** Access the original repository [here](https://github.com/Azure/azure-sdk-for-python).

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features:

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, from storage and compute to databases and AI.
*   **Modular Libraries:**  Utilize individual service-specific libraries for efficient development and reduced package size.
*   **Modern Design:** Benefit from libraries built using the latest [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), offering a consistent and intuitive experience.
*   **Simplified Authentication:** Seamlessly integrate with Azure Active Directory and other authentication methods.
*   **Cross-Platform Compatibility:**  Supported on Python 3.9 and later, ensuring broad compatibility.
*   **Management Libraries:** Provision and manage Azure resources programmatically with dedicated management libraries (e.g., `azure-mgmt-compute`).
*   **Extensive Documentation:**  Access comprehensive documentation, code samples, and migration guides to accelerate your development.
*   **Telemetry Control:**  Configure and opt-out of telemetry collection to meet your specific needs.

## Getting Started

The Azure SDK for Python is designed for ease of use.  Each service has a dedicated library.  To get started, navigate to the `/sdk` directory to find available service libraries. Refer to the `README.md` (or `README.rst`) file within each library's project folder for detailed instructions.

### Prerequisites

Ensure you have Python 3.9 or later installed. Review the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.

## Packages Available

The SDK offers a variety of packages, categorized as follows:

*   **Client - New Releases:** GA (General Availability) and Preview libraries for interacting with existing Azure resources. [Most up-to-date list](https://azure.github.io/azure-sdk/releases/latest/index.html#python)
*   **Client - Previous Versions:** Stable, production-ready versions of client libraries.
*   **Management - New Releases:**  Libraries adhering to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), offering core capabilities and intuitive management features. [Most up-to-date list](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)
*   **Management - Previous Versions:** Libraries for provisioning and managing Azure resources. [Complete list](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Identified by `azure-mgmt-` namespace.

> **Note:**  For production environments, prioritize stable, non-preview libraries.  Consult the migration guide if you are upgrading management libraries [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:**  Search or ask questions on StackOverflow using the `azure` and `python` tags.

## Data Collection and Telemetry

The SDK collects data to improve services.  Review Microsoft's [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) for details.

### Telemetry Configuration

Telemetry is enabled by default.  To opt-out, create a `NoUserAgentPolicy` class, a subclass of `UserAgentPolicy`, and pass it to client construction.

```python
import os
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient
from azure.core.pipeline.policies import UserAgentPolicy

# Create your credential
mi_credential = ManagedIdentityCredential()
account_url = "https://<storageaccountname>.blob.core.windows.net"

# Set up user-agent override
class NoUserAgentPolicy(UserAgentPolicy):
    def on_request(self, request):
        pass

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient(account_url, credential=mi_credential, user_agent_policy=NoUserAgentPolicy())
container_client = blob_service_client.get_container_client(container=<container_name>) 
```

## Reporting Security Issues

Report security issues to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details.
This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).