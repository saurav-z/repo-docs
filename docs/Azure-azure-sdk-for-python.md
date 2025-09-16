# Azure SDK for Python: Simplify Cloud Development

**Build robust and scalable Python applications with the official Azure SDK, offering comprehensive libraries for interacting with Azure services.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services through dedicated Python libraries.
*   **Latest Releases & Previews:** Stay up-to-date with the newest GA (Generally Available) and preview packages.
*   **Management Libraries:** Provision and manage Azure resources with the Azure SDK Design Guidelines.
*   **Production-Ready Stability:** Leverage stable, non-preview libraries for production environments.
*   **Easy to Get Started:**  Each service offers separate libraries that you can choose to use, instead of one large package.

## Getting Started

Get started using the appropriate `README.md` or `README.rst` files located in each library's project folder within the `/sdk` directory.

### Prerequisites

The client libraries support Python 3.9 or later.  See the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.

## Available Packages

Choose from a variety of packages categorized by service and release status:

*   **Client: New Releases:**  GA and preview libraries that offer essential functionalities for consuming existing resources (e.g., uploading blobs). These releases share core features like retry mechanisms, logging, and authentication.  [Latest Package List](https://azure.github.io/azure-sdk/releases/latest/index.html#python)
*   **Client: Previous Versions:** Stable, production-ready versions of packages for interacting with Azure services. These may have slightly different feature sets than new releases.
*   **Management: New Releases:**  Libraries following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) providing core capabilities, including the Azure Identity library, an HTTP Pipeline with custom policies, error-handling, distributed tracing, and much more.  [Documentation and Code Samples](https://aka.ms/azsdk/python/mgmt).  [Migration Guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide)
*   **Management: Previous Versions:**  Comprehensive management libraries for provisioning and managing Azure resources. [Complete Package List](https://azure.github.io/azure-sdk/releases/latest/all/python.html)

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:**  Search or ask questions on [StackOverflow](https://stackoverflow.com/questions/tagged/azure+python) using the `azure` and `python` tags.

## Data Collection and Telemetry

The SDK collects usage data for service improvement. You can learn more in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704).  For specific details on collected data, see the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is enabled by default. To opt out, disable telemetry at client construction using a custom `NoUserAgentPolicy`.

Example:
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

Report security vulnerabilities privately via email to the Microsoft Security Response Center (MSRC): <secure@microsoft.com>. You should receive a response within 24 hours.  See the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue) for further information.

## Contributing

Contributions are welcome! Please see the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details. All contributions require a Contributor License Agreement (CLA).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.