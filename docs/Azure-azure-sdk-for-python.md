# Azure SDK for Python: Simplify Your Cloud Development

**Accelerate your Python cloud development with the official Azure SDK, offering robust libraries for seamless integration with Azure services.**

[Link to Original Repo: Azure/azure-sdk-for-python](https://github.com/Azure/azure-sdk-for-python)

## Key Features

*   **Comprehensive Service Coverage:** Access a wide array of Azure services, including storage, compute, databases, and more.
*   **Modular Design:** Choose individual service libraries for a lightweight and focused development experience.
*   **Consistent API Design:** Benefit from a unified and intuitive API across all Azure services, making it easy to learn and use.
*   **Production-Ready:** Leverage stable, well-tested libraries for your production environments, with preview releases available for early access to new features.
*   **Modern Authentication:** Securely authenticate with Azure services using the latest authentication protocols.
*   **Azure SDK Design Guidelines:** Benefit from packages that align with the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/).

## Getting Started

Each Azure service has its own dedicated library, offering a streamlined approach to development. For detailed instructions on getting started, consult the `README.md` (or `README.rst`) file within each service's project folder, located in the `/sdk` directory.

## Prerequisites

The Azure SDK for Python supports Python 3.9 and later. For detailed information, please consult the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Available Packages

The SDK offers various packages categorized by service and release status:

*   **Client - New Releases:** Utilize the latest GA and preview libraries for interacting with Azure services (e.g., uploading a blob).  Find the most up-to-date list [here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).
*   **Client - Previous Versions:** Leverage stable, production-ready packages for wider service coverage and compatibility.
*   **Management - New Releases:** Discover new management libraries that follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), providing capabilities such as Azure Identity and error handling. Find the latest list [here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html). See the [migration guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide) for updating from older versions.
*   **Management - Previous Versions:**  Access management libraries that allow you to provision and manage Azure resources. Browse available libraries [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html).

## Need Help?

*   **Documentation:** Detailed documentation is available at [Azure SDK for Python documentation](https://aka.ms/python-docs).
*   **Issues:** Report issues through [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues).
*   **Community Support:** Find answers on StackOverflow using the tags `azure` and `python`.

## Data Collection

The software may collect usage data that is sent to Microsoft. You can disable telemetry by using `NoUserAgentPolicy()` as detailed in the example below:

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

Report security issues privately to the Microsoft Security Response Center (MSRC) via <secure@microsoft.com>.

## Contributing

Contribute to the project by following the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md). This project welcomes contributions and suggestions and requires a Contributor License Agreement (CLA). This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).