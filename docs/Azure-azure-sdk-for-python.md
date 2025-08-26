# Azure SDK for Python: Simplify Cloud Development

**Build robust and scalable applications for Microsoft Azure with the official Azure SDK for Python.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

This SDK provides comprehensive libraries for interacting with various Azure services, making it easier than ever to integrate cloud capabilities into your Python projects.

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, databases, AI, and more.
*   **Consistent Design & APIs:** Benefit from a unified design and consistent API across all Azure services, simplifying learning and usage.
*   **Modern Authentication:** Leverage the latest authentication methods for secure and seamless access to Azure resources.
*   **Robust Error Handling:** Built-in retry mechanisms, logging, and error handling for improved application reliability.
*   **Active Development & Support:** Benefit from ongoing updates, new features, and comprehensive documentation and support.
*   **Modular Design:** Choose the specific service libraries you need, minimizing dependencies and improving performance.
*   **Version Support Policy:** Follows a clear version support policy, ensuring stability and long-term maintainability.

## Getting Started

*   **Service-Specific Libraries:** Each Azure service has its own Python library. Explore the `/sdk` directory for available libraries.
*   **Prerequisites:** Requires Python 3.9 or later.
*   **Documentation:** Access comprehensive documentation for each library via the [public developer docs](https://docs.microsoft.com/python/azure/) or the versioned [developer docs](https://azure.github.io/azure-sdk-for-python).

## Packages Available

The Azure SDK for Python provides client and management libraries, categorized as:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the latest **GA (Generally Available)** and **preview** packages, offering new features and functionalities.  They adhere to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/index.html) and share core functionalities like retries, logging, and authentication.

*   [Up-to-date list of new packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python)
    > **Note:** Use stable, non-preview libraries for production environments.

### Client: Previous Versions

These are the last stable versions of production-ready packages providing similar functionalities, but may not follow the latest guidelines or have all the features of the new releases. They generally offer wider coverage of services.

### Management: New Releases

These libraries are also built with the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) in mind and provide a number of core capabilities that are shared amongst all Azure SDKs, including the intuitive Azure Identity library, an HTTP Pipeline with custom policies, error-handling, distributed tracing, and much more.

*   [Documentation and code samples](https://aka.ms/azsdk/python/mgmt)
*   [Migration Guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide)
*   [Up-to-date list of all new packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

> **Note:** Use stable, non-preview libraries for production. Refer to the migration guide if you upgraded to new SDK versions to avoid authentication issues.

### Management: Previous Versions

Complete list of management libraries for provisioning and managing Azure resources: [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Identify management libraries by namespaces starting with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   [Azure SDK for Python Documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   [Stack Overflow](https://stackoverflow.com/questions/tagged/azure+python) (use tags `azure` and `python`)

## Data Collection & Telemetry

The SDK collects usage data to improve services.

*   Learn more in the [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).
*   Telemetry is enabled by default.  You can disable it when creating a client using a custom `NoUserAgentPolicy`.

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

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

*   Further information is available in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details.

*   Contributions require a Contributor License Agreement (CLA).
*   This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.