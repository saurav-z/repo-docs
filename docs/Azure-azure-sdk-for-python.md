# Azure SDK for Python: Simplify Cloud Development

**Get started with the Azure SDK for Python and build robust, scalable applications that seamlessly integrate with Microsoft Azure services.** ([View the Original Repository](https://github.com/Azure/azure-sdk-for-python))

The Azure SDK for Python provides a comprehensive set of libraries for interacting with Azure services, simplifying cloud development and accelerating your projects.

**Key Features:**

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including compute, storage, databases, networking, and more.
*   **Modern Design:** Utilize libraries built with the latest Azure SDK Design Guidelines, ensuring consistency and ease of use.
*   **Enhanced Authentication:** Leverage the intuitive Azure Identity library for secure and streamlined authentication.
*   **Shared Core Functionality:** Benefit from shared features like retries, logging, transport protocols, and authentication protocols in the `azure-core` library.
*   **Well-Defined Guidelines:** Follow the Azure SDK Design Guidelines for Python to create production-ready code.
*   **Up-to-date packages:** Regularly updated with GA and preview releases to support the latest Azure services and features.

## Getting Started

Easily integrate specific Azure services into your Python applications by using separate libraries. Each service library has its own `README.md` or `README.rst` file that contains information to get you started.

### Prerequisites

The client libraries support Python 3.9 and later.  See the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.

## Available Packages

The Azure SDK for Python offers a range of packages categorized for different needs:

*   **Client - New Releases:** (GA and Preview) New packages for using and consuming existing Azure resources.  Find the [latest list](https://azure.github.io/azure-sdk/releases/latest/index.html#python).
*   **Client - Previous Versions:** Stable, production-ready versions for wider service coverage.
*   **Management - New Releases:** Management libraries that follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for provisioning and managing resources. Find the [latest list](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).
*   **Management - Previous Versions:** Management libraries for provisioning and managing resources, identified by `azure-mgmt-` namespaces.

## Need Help?

*   **Documentation:** Explore detailed documentation at [Azure SDK for Python documentation](https://aka.ms/python-docs).
*   **Issues:** Report issues via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues).
*   **Community:** Search [previous questions](https://stackoverflow.com/questions/tagged/azure+python) or ask new ones on StackOverflow using the `azure` and `python` tags.

## Data Collection

The Azure SDK for Python collects telemetry data to improve its products and services. You can learn more about data collection and usage in the [Microsoft privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry collection is enabled by default. You can opt out by disabling it during client construction.

**Example:**

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

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.  Get more information in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

Contributions are welcome! Review the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details.  All contributions require agreement to a Contributor License Agreement (CLA).  This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).