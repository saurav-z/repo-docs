# Azure SDK for Python: Simplify Cloud Development

**The Azure SDK for Python provides a comprehensive set of libraries for developers to interact with and manage Azure services, offering a streamlined and efficient cloud development experience.** Explore the complete suite of tools and start building with confidence! [View the original repository](https://github.com/Azure/azure-sdk-for-python).

## Key Features of the Azure SDK for Python

*   **Comprehensive Service Coverage:** Access a vast array of Azure services with dedicated client libraries, covering everything from storage and compute to databases and AI.
*   **Modern Design Guidelines:** Benefit from libraries built according to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), ensuring consistency, ease of use, and a familiar developer experience.
*   **Authentication and Security:** Leverage secure authentication mechanisms through the Azure Identity library.
*   **Client and Management Libraries:** Utilize both client libraries for interacting with existing resources and management libraries for provisioning and managing Azure infrastructure.
*   **Regular Updates & Support:** Stay current with the latest Azure features and improvements, with ongoing updates and detailed documentation.

## Getting Started

Each Azure service has its own set of Python libraries. To get started:

1.  **Explore the `/sdk` directory:** Find service-specific libraries within the `/sdk` directory.
2.  **Review `README.md` files:** Each library's `README.md` (or `README.rst`) file contains detailed instructions for usage.

### Prerequisites

*   The client libraries are supported on Python 3.9 or later.
*   For further details, please see the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

The Azure SDK for Python is organized into several categories.

*   **Client - New Releases:** Modern, GA and preview packages for interacting with resources (e.g., upload a blob).
*   **Client - Previous Versions:** Stable, production-ready versions of packages.
*   **Management - New Releases:** New management libraries that follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/).
*   **Management - Previous Versions:** Older versions of management libraries, offering wider service coverage.

**Note:**  Always prefer stable, non-preview libraries for production environments.

### Links to Package Lists:

*   [Client Packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python)
*   [Management Packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)
*   [All Packages](https://azure.github.io/azure-sdk/releases/latest/all/python.html)

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Stack Overflow:** Search or ask questions using the `azure` and `python` tags.

## Data Collection and Telemetry

The SDK collects data to improve its services. You can opt out of telemetry using the provided example code and `NoUserAgentPolicy` class, as detailed in the original README. [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy)

## Security

Report security issues to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

Contributions are welcome! See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details.