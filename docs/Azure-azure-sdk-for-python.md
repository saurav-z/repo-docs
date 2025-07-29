# Azure SDK for Python: Simplify Cloud Development

**Empower your Python applications with the official Azure SDK, providing comprehensive libraries to interact with Azure services effectively.** [Explore the original repository.](https://github.com/Azure/azure-sdk-for-python)

This repository hosts the active development of the Azure SDK for Python. It offers a rich set of Python libraries designed to simplify the development of applications that interact with Microsoft Azure cloud services. Whether you're building applications for storage, compute, databases, or other Azure services, this SDK provides the tools you need.

## Key Features:

*   **Comprehensive Service Coverage:** Access a wide range of Azure services through dedicated client libraries.
*   **Modern Design:** Follows the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) to ensure consistency and ease of use.
*   **Enhanced Functionality:** Benefit from features like retries, logging, authentication, and HTTP pipeline support.
*   **Management Libraries:**  Manage Azure resources programmatically using management libraries that streamline provisioning and configuration.
*   **Easy to Get Started:** Each service has its own set of libraries, with clear documentation and getting started guides within the respective library's `README.md` (or `README.rst`) file.
*   **Version Support Policy:** Supports Python 3.9 and later, with clear documentation on [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Getting Started

To begin using the Azure SDK for Python, select the specific service you need and refer to the corresponding `README.md` or `README.rst` file within the `/sdk` directory for detailed instructions and code samples.

## Package Availability

The Azure SDK for Python provides a range of libraries, categorized as follows:

*   **Client: New Releases:**  Generally Available (GA) and preview packages that enable interaction with existing Azure resources. [Find the most up-to-date list.](https://azure.github.io/azure-sdk/releases/latest/index.html#python)
*   **Client: Previous Versions:** Stable, production-ready libraries for interacting with Azure services.
*   **Management: New Releases:**  Modern management libraries, following Azure SDK Design Guidelines. [View documentation and samples.](https://aka.ms/azsdk/python/mgmt) [Check the most up to date list.](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)
*   **Management: Previous Versions:**  Libraries for provisioning and managing Azure resources. [Find a complete list.](https://azure.github.io/azure-sdk/releases/latest/all/python.html)

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   StackOverflow ([azure](https://stackoverflow.com/questions/tagged/azure) and [python](https://stackoverflow.com/questions/tagged/python) tags)

## Data Collection

The SDK collects telemetry data to improve services. You can opt out by disabling telemetry during client construction using a custom `UserAgentPolicy`. [See the Telemetry Configuration section for details.](https://github.com/Azure/azure-sdk-for-python#telemetry-configuration)  Learn more in the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

## Reporting Security Issues

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

Contributions are welcome!  See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details.