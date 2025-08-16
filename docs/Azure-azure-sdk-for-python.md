# Azure SDK for Python: Build Robust Cloud Applications

**Empower your Python projects with the official Azure SDK, enabling seamless integration with a comprehensive suite of Microsoft Azure services.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

## Key Features:

*   **Comprehensive Service Coverage:** Access a wide array of Azure services, including storage, compute, databases, AI, and more.
*   **Client and Management Libraries:** Utilize distinct libraries for interacting with existing resources (clients) and managing Azure infrastructure (management).
*   **Consistent API Design:** Benefit from standardized design guidelines across libraries, ensuring a predictable and intuitive developer experience.
*   **Production-Ready Libraries:** Leverage stable, non-preview libraries for reliable and production-ready implementations.
*   **Azure SDK Design Guidelines for Python:** Implement the most current best practices for your Python projects

## Getting Started

Each Azure service has dedicated Python libraries, offering granular control and streamlined development. Explore the `README.md` (or `README.rst`) files within the respective service folders in the `/sdk` directory to get started.

### Prerequisites

The client libraries support Python 3.9 and later. For further details, review the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Package Categories:

*   **Client - New Releases:** Cutting-edge packages currently in GA (General Availability) or preview, providing access to existing resources.
*   **Client - Previous Versions:** Stable, production-ready versions offering similar functionality to the new releases.
*   **Management - New Releases:** Management libraries adhering to Azure SDK Design Guidelines, for provisioning and managing Azure resources. [Documentation](https://aka.ms/azsdk/python/mgmt) and [Migration Guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).
*   **Management - Previous Versions:** Comprehensive management libraries enabling Azure resource provisioning and management.

## Need Help?

*   **Documentation:** Access in-depth documentation at [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **GitHub Issues:** Report issues and provide feedback through [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Stack Overflow:** Search or ask questions using the `azure` and `python` tags on Stack Overflow ([previous questions](https://stackoverflow.com/questions/tagged/azure+python))

## Data Collection

The SDK collects telemetry data to improve services. You can opt out of telemetry by disabling it during client construction.
Refer to the [Telemetry Configuration](https://github.com/Azure/azure-sdk-for-python/blob/main/README.md#telemetry-configuration) section in the original README for specific implementation instructions. For information on data collection, visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

## Reporting Security Issues

Report security vulnerabilities privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

Contribute to the project by following the guidelines outlined in the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md). This project uses the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).