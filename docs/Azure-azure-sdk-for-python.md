# Azure SDK for Python: Simplify Cloud Development

**Build robust and scalable applications for Microsoft Azure with the official Azure SDK for Python.** This comprehensive library provides a unified and consistent experience for interacting with a wide range of Azure services.  ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

## Key Features:

*   **Comprehensive Service Coverage:** Access a vast array of Azure services, including storage, compute, networking, databases, and more.
*   **Consistent API Design:** Benefit from a unified approach to authentication, error handling, and other core functionalities across all services.
*   **High Performance & Reliability:** Built for production use, with features like retry mechanisms and optimized transport protocols.
*   **Modern Python Support:** Compatible with Python 3.9 and later.
*   **Client & Management Libraries:** Utilize client libraries for interacting with existing resources and management libraries for provisioning and managing Azure resources.
*   **Well-Documented & Supported:**  Benefit from extensive documentation, code samples, and community support.
*   **Telemetry Configuration:** Opt-out of telemetry data collection for privacy or compliance reasons.

## Getting Started

Choose from a range of individual service libraries to get started.  Find `README.md` or `README.rst` files within each service's project folder in the `/sdk` directory for specific instructions.

### Prerequisites

Ensure you have Python 3.9 or later installed.  See the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for detailed version support information.

## Package Categories

*   **Client Libraries (New Releases):**  **GA** (Generally Available) and **preview** packages for consuming and interacting with Azure resources (e.g., uploading blobs).  They include core functionalities like retries, logging, authentication, etc. Find the latest packages [here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).
    *   **Important:** Use stable, non-preview libraries for production readiness.
*   **Client Libraries (Previous Versions):** Production-ready, stable versions offering similar functionalities to preview releases. May not fully adhere to the latest guidelines or offer the same feature set but offer broader service coverage.
*   **Management Libraries (New Releases):**  Provides libraries that follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). Learn more and find samples [here](https://aka.ms/azsdk/python/mgmt).  Review the migration guide [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide) if upgrading. Find the newest management libraries [here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)
    *   **Important:** Use stable, non-preview libraries for production readiness. Review the migration guide if you have authentication issues after upgrades.
*   **Management Libraries (Previous Versions):**  Enable provisioning and managing Azure resources.  A complete list is available [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html).  Management libraries are identified by namespaces starting with `azure-mgmt-`, e.g., `azure-mgmt-compute`.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:**  [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:** Search [StackOverflow](https://stackoverflow.com/questions/tagged/azure+python) with `azure` and `python` tags.

## Data Collection

Microsoft collects data about your use of the software to improve services.  Learn more in the [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is on by default, but you can disable it. To opt-out, define a `NoUserAgentPolicy` and pass an instance of it during client creation:

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

Report security vulnerabilities to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.  This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).