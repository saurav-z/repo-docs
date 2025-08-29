# Azure SDK for Python

**Empower your Python applications with seamless access to a wide range of Azure services using the official Azure SDK for Python!** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository hosts the actively developed Azure SDK for Python, providing a robust and consistent experience for interacting with Azure services.

## Key Features

*   **Comprehensive Service Coverage:** Access a vast array of Azure services, including storage, compute, networking, databases, and more.
*   **Consistent API Design:**  Benefit from a unified API design across all Azure services, making development easier and more efficient.
*   **Modern Authentication:**  Leverage the latest authentication methods, including Azure Active Directory and managed identities.
*   **Reliable and Performant:** Designed for production use, offering features such as retries, logging, and efficient transport protocols.
*   **Regular Updates:** Stay up-to-date with the latest Azure service features and improvements through frequent package releases.
*   **Management & Client Libraries:** Utilize new and previous versions of Client & Management libraries designed to create, consume, and interact with Azure resources.

## Getting Started

Each Azure service has a dedicated set of Python libraries, making it easy to incorporate only the necessary components into your project.  Find service-specific libraries within the `/sdk` directory.

### Prerequisites

The client libraries are compatible with Python 3.9 or later. For detailed information on version support, see the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

The SDK offers various libraries categorized by function and version:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the newest libraries, generally **GA** (Generally Available) or in **preview**.  They allow you to use and consume existing resources, offering core functionalities like retries, logging, and authentication.

*   Find the [most up-to-date package list here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).
>   **Note:** Use stable, non-preview libraries for production readiness.

### Client: Previous Versions

These are the last stable versions, offering a production-ready option to access services with similar functionalities as the Preview releases, but they may not implement the latest guidelines.
### Management: New Releases
A set of management libraries that follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) are now available. These new libraries provide a number of core capabilities that are shared amongst all Azure SDKs, including the intuitive Azure Identity library, an HTTP Pipeline with custom policies, error-handling, distributed tracing, and much more.
Documentation and code samples for these new libraries can be found [here](https://aka.ms/azsdk/python/mgmt). In addition, a migration guide that shows how to transition from older versions of libraries is located [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

You can find the [most up to date list of all of the new packages on our page](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

>   **Note:** Use stable, non-preview libraries for production readiness. Also, if you are experiencing authentication issues with the management libraries after upgrading certain packages, it's possible that you upgraded to the new versions of SDK without changing the authentication code, please refer to the migration guide mentioned above for proper instructions.

### Management: Previous Versions

These libraries provision and manage Azure resources.
*   For a complete list, please [check here](https://azure.github.io/azure-sdk/releases/latest/all/python.html).
*   Management libraries are identified by namespaces starting with `azure-mgmt-`, e.g., `azure-mgmt-compute`.

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   [Stack Overflow](https://stackoverflow.com/questions/tagged/azure+python) using `azure` and `python` tags.

## Data Collection and Telemetry

This software collects data to improve services. Learn more in the [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is enabled by default. To disable it, you can pass a custom policy during client creation.

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

Report security issues to the Microsoft Security Response Center (MSRC) <secure@microsoft.com>.

## Contributing

Contribute by following the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) and agreeing to the Contributor License Agreement (CLA).