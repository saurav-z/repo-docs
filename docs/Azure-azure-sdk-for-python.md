# Azure SDK for Python: Powerful Libraries for Cloud Development

**Build robust and scalable applications with ease using the official Azure SDK for Python.** [Visit the original repository](https://github.com/Azure/azure-sdk-for-python)

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features:

*   **Comprehensive Service Coverage:** Access a wide range of Azure services with dedicated libraries for client and management operations.
*   **Modern Design:**  Leverage the latest Azure SDK Design Guidelines for Python, including intuitive Azure Identity, HTTP Pipeline, and more.
*   **Simplified Authentication:** Seamlessly integrate with Azure Active Directory and other authentication methods.
*   **Cross-Platform Compatibility:**  Support for Python 3.9 and later, ensuring broad compatibility.
*   **Reliable and Stable Libraries:** Utilize GA (General Availability) packages for production-ready applications, along with the availability of previous versions.
*   **Extensive Documentation and Samples:** Get started quickly with comprehensive documentation and code samples.
*   **Telemetry Configuration:** Easily opt-out of telemetry collection to comply with your privacy standards.
*   **Security Focus:** Dedicated procedures for reporting security issues, and compliance with Microsoft's Security Response Center (MSRC) standards.

## Getting Started

Explore the library's `README.md` (or `README.rst`) file within each service's project folder for service-specific setup and usage instructions. Service libraries are located within the `/sdk` directory.

## Available Packages:

*   **Client Libraries (New Releases):**  Focus on interacting with existing resources, offering features like retries, logging, and authentication, following [Azure SDK Design Guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html).  Find the [latest packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python).
    *   **Note:**  Use stable, non-preview libraries for production readiness.
*   **Client Libraries (Previous Versions):**  Stable, production-ready versions with similar functionality.
*   **Management Libraries (New Releases):**  Management libraries that follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) that provide infrastructure provisioning and management features.  Find the [latest packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).
    *   **Note:**  Consult the [migration guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide) when upgrading to new management library versions.
*   **Management Libraries (Previous Versions):**  Libraries for provisioning and managing Azure resources. These can be identified by namespaces that start with `azure-mgmt-`, e.g. `azure-mgmt-compute`.  Explore the [complete list](https://azure.github.io/azure-sdk/releases/latest/all/python.html).

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   [StackOverflow](https://stackoverflow.com/questions/tagged/azure+python) (use tags `azure` and `python`)

## Data Collection and Telemetry

Learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is enabled by default. To disable telemetry, use the provided code example using `NoUserAgentPolicy`:

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

Report security issues privately to the [Microsoft Security Response Center (MSRC)](mailto:secure@microsoft.com).

## Contributing

Contribute to this project by following the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) and agreeing to the Contributor License Agreement (CLA).  See the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) for questions.