# Azure SDK for Python: Build Robust Cloud Applications

**Unlock the power of the Microsoft Azure cloud with the Azure SDK for Python, a comprehensive library designed to simplify your cloud development journey.** Learn more about this SDK on its [GitHub repository](https://github.com/Azure/azure-sdk-for-python).

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features:

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, from storage and compute to databases and AI.
*   **Simplified Development:**  Easily interact with Azure resources using intuitive Python libraries.
*   **Modern Design:** Benefit from libraries built with the latest Azure SDK Design Guidelines for Python.
*   **Asynchronous Support:** Leverage asynchronous operations for improved performance and scalability.
*   **Production-Ready Libraries:** Utilize stable, non-preview libraries for reliable production deployments.
*   **Management Capabilities:** Provision and manage your Azure resources with dedicated management libraries.
*   **Telemetry Control:**  Configure telemetry settings to control data collection.

## Getting Started

Each Azure service has dedicated Python libraries. To get started:

1.  **Explore Service Libraries:** Find libraries in the `/sdk` directory.
2.  **Review `README.md`:** Consult the `README.md` (or `README.rst`) file within each library's project folder for specific instructions.
3.  **Prerequisites:** Ensure you have Python 3.9 or later.  See the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for more details.

## Packages Available

The SDK is organized into distinct categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries allow you to use and consume existing resources, sharing core functionalities like retries, logging, and authentication. The most up-to-date list of packages is available [here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** For production environments, prioritize stable, non-preview libraries.

### Client: Previous Versions

These are stable versions of packages for production use, offering similar functionalities to new releases with potentially wider service coverage.

### Management: New Releases

Management libraries follow Azure SDK Design Guidelines. Documentation and code samples are available [here](https://aka.ms/azsdk/python/mgmt).  A migration guide is located [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide). The most up-to-date list of packages is available [here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> **Note:**  If experiencing authentication issues after upgrading, consult the migration guide.

### Management: Previous Versions

For a complete list of management libraries to provision and manage Azure resources, [check here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). These libraries can be identified by namespaces starting with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:**  Search or ask on StackOverflow using the `azure` and `python` tags.

## Data Collection

The software collects information to improve products and services.  You can find more information in the help documentation, Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704), and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is on by default. To opt-out, disable telemetry at client construction.

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

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing. All contributions require a Contributor License Agreement (CLA).

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.