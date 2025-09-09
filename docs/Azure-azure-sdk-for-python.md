# Azure SDK for Python: Build Robust Applications on Azure

**Empower your Python applications with the official Azure SDK, providing comprehensive libraries for interacting with Microsoft Azure services.**  Explore the comprehensive [Azure SDK for Python](https://github.com/Azure/azure-sdk-for-python) and its offerings.

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, networking, databases, and more.
*   **Modular Design:** Utilize individual, service-specific libraries for focused development and efficient resource usage.
*   **Consistent API:** Enjoy a unified and intuitive programming experience across different Azure services, simplifying learning and adoption.
*   **Client and Management Libraries:** Leverage client libraries for direct interaction with Azure resources and management libraries for provisioning and configuration.
*   **Latest Updates:** Receive regular updates, including new features and improvements, to ensure optimal performance and access to the latest Azure capabilities.

## Getting Started

*   **Choose Your Library:** Select the specific library corresponding to the Azure service you wish to use.
*   **Explore Examples:** Refer to the `README.md` (or `README.rst`) file within each library's project folder for detailed usage examples.
*   **Prerequisites:** Ensure you have Python 3.9 or later installed.

## Available Packages

Find service libraries in the `/sdk` directory. Each service provides multiple libraries, categorized as:

*   **Client - New Releases:** GA (Generally Available) and preview libraries. Interact with existing resources, such as uploading a blob.
*   **Client - Previous Versions:** Stable, production-ready versions of packages, providing a wide coverage of services.
*   **Management - New Releases:** Libraries following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), which allow you to provision and manage Azure resources.
*   **Management - Previous Versions:** Complete list of management libraries for provisioning and managing Azure resources.

### Important Notes

*   **Production Readiness:**  For production use, leverage stable, non-preview libraries.
*   **Authentication for Management Libraries:** Refer to the [migration guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide) for upgrading to new versions and authentication issues.

## Need Help?

*   **Documentation:** Access detailed documentation at [Azure SDK for Python documentation](https://aka.ms/python-docs).
*   **Community Support:**
    *   File issues on [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues).
    *   Search and ask questions on Stack Overflow using the `azure` and `python` tags.

## Data Collection and Telemetry

The Azure SDK collects telemetry data to improve performance and services. You can opt-out of telemetry.

### Telemetry Configuration

Telemetry collection is on by default. To opt-out, disable telemetry at client construction by creating a `NoUserAgentPolicy` and passing it as `user_agent_policy` when creating the client:

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

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing to this project.

This project welcomes contributions and follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).