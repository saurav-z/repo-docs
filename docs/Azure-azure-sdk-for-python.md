# Azure SDK for Python: Empowering Python Developers with the Cloud

The **Azure SDK for Python** provides a comprehensive suite of libraries, empowering Python developers to seamlessly integrate and manage Azure cloud services within their applications.  ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

## Key Features:

*   **Broad Coverage:** Access a wide range of Azure services, including compute, storage, networking, databases, and more.
*   **Modular Design:** Utilize individual, service-specific libraries for focused development and minimal dependencies.
*   **Cross-Platform Compatibility:**  Supported on Python 3.9 and later, enabling deployment across various environments.
*   **Client Libraries:** Leverage new releases (GA and preview) to consume existing resources and interact with them.
*   **Management Libraries:** Provision and manage Azure resources with libraries that follow the Azure SDK Design Guidelines for Python.
*   **Core Functionality:** Benefit from shared functionalities like retries, logging, authentication, and HTTP pipeline features.
*   **Comprehensive Documentation:** Access detailed documentation and code samples to get started quickly.
*   **Active Development:** Benefit from ongoing development and updates to ensure the latest features and best practices.

## Getting Started

Each Azure service has a dedicated Python library. To begin, explore the `README.md` (or `README.rst`) files within the library's project folder.

### Prerequisites
Ensure you have Python 3.9 or later installed.  Refer to the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for detailed information.

## Packages Available

The Azure SDK for Python offers various libraries categorized as follows:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These packages are announced as General Availability (GA) or are currently in preview, allowing you to use and consume existing resources. Find the [latest packages here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).
**Note:**  For production environments, use stable, non-preview libraries.

### Client: Previous Versions

These are the last stable, production-ready versions of client libraries that provide comprehensive service coverage.

### Management: New Releases

New management libraries are available, adhering to [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). Find the [latest packages here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html). Includes an intuitive Azure Identity library, an HTTP Pipeline with custom policies, error-handling, and more.
**Note:**  Refer to the migration guide ([Migration Guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide)) if you're upgrading and experiencing authentication issues.

### Management: Previous Versions

For a complete list of management libraries, see: [Complete Management Libraries](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Management libraries have namespaces starting with `azure-mgmt-`.

## Need Help?

*   [Azure SDK for Python Documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   Stack Overflow ([`azure` and `python` tags](https://stackoverflow.com/questions/tagged/azure+python))

## Data Collection

The software may collect usage data. See the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) and the [Microsoft privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) for details.

### Telemetry Configuration

Telemetry is enabled by default.  To disable it:

1.  Define a `NoUserAgentPolicy` class that inherits from `UserAgentPolicy` and overrides the `on_request` method to do nothing.
2.  Pass an instance of this class as `user_agent_policy=NoUserAgentPolicy()` during client creation.

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

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details.  This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).