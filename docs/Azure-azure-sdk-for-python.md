# Azure SDK for Python: Build Robust Python Applications on Azure

This comprehensive SDK empowers Python developers to seamlessly integrate with and manage Azure cloud services.  [Explore the original repository](https://github.com/Azure/azure-sdk-for-python).

**Key Features:**

*   **Broad Service Coverage:** Access a wide array of Azure services, including storage, compute, databases, and more.
*   **Client & Management Libraries:** Utilize both client libraries for interacting with existing resources and management libraries for provisioning and managing Azure infrastructure.
*   **Up-to-Date Packages:** Stay current with the latest releases and preview features, as well as stable, production-ready packages.
*   **Azure SDK Design Guidelines:** Adherence to consistent design principles ensures a familiar and intuitive development experience.
*   **Comprehensive Documentation:** Access detailed documentation, including quickstarts and migration guides.
*   **Open Source & Contributing:** Contribute to the project and benefit from the open-source community.

**Available Packages:**

*   **Client Libraries (New Releases):**  GA and Preview libraries for using and consuming existing resources (e.g., uploading blobs).
*   **Client Libraries (Previous Versions):** Stable, production-ready versions offering a broad service coverage.
*   **Management Libraries (New Releases):**  Libraries for provisioning and managing Azure resources, following the Azure SDK Design Guidelines.
*   **Management Libraries (Previous Versions):**  Libraries that enable you to provision and manage Azure resources.

**Getting Started:**

*   Python 3.9 or later is required.
*   Explore the `/sdk` directory for service-specific libraries.
*   Refer to the `README.md` (or `README.rst`) in each library's project folder for specific instructions.
*   Find the most up-to-date package lists [here](https://azure.github.io/azure-sdk/releases/latest/index.html#python) and [here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

**Need Help?**

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   StackOverflow: Use tags `azure` and `python`.

**Data Collection & Telemetry Configuration:**
The SDK collects telemetry data by default. You can opt-out by defining a `NoUserAgentPolicy` class and passing it as `user_agent_policy` during client creation.

**Reporting Security Issues:**  Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

**Contributing:**  Review the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) and contribute to the project.