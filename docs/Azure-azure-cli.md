# Azure CLI: Manage Your Azure Resources from the Command Line

**Simplify Azure cloud management with the Azure CLI, a powerful, cross-platform command-line tool for interacting with your Azure services.**  ([Back to Original Repo](https://github.com/Azure/azure-cli))

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.
*   **Comprehensive Azure Service Support:** Manage a vast array of Azure services, including virtual machines, storage, networking, and more.
*   **Intuitive Command Structure:** Easy-to-learn `az [group] [subgroup] [command] {parameters}` syntax.
*   **Tab Completion:** Enhance productivity with tab completion for commands, groups, and parameters.
*   **Powerful Querying with JMESPath:** Customize output and extract specific data using the `--query` parameter and JMESPath.
*   **Scripting-Friendly Exit Codes:**  Leverage standardized exit codes for automation and scripting.
*   **VS Code Integration:** Enhance your workflow with the Azure CLI Tools extension for Visual Studio Code, including IntelliSense, snippets, and integrated terminal support.
*   **Flexible Output Formatting:** Supports JSON, table, and TSV output formats to suit your needs.

## Getting Started

*   **Installation:**  Follow the detailed installation instructions in the [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli). Troubleshoot common issues with the [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).
*   **Cloud Shell:** Get started quickly with the Azure CLI using [Azure Cloud Shell](https://portal.azure.com/#cloudshell).
*   **Basic Usage:**
    *   Use `az [group] [subgroup] [command] {parameters}` for command execution.
    *   Get help with the `-h` parameter (e.g., `az storage -h`).
    *   Explore more examples in the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) or [https://learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview).
    *   Learn how to use the CLI effectively in the guide [Tips for using Azure CLI effectively](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively).

## Developer Installation and Contribution

*   **Developer Installation:**
    *   **Docker:** Preconfigured Docker image available. See [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list)
    *   **Edge Builds:** Access the latest builds from the `dev` branch for testing and early access.  See links in the original README to install the latest edge builds.
    *   **Get builds of arbitrary commit or PR**  See: [Try new features before release](doc/try_new_features_before_release.md)
    *   **Developer Setup:**
        *   Configure your machine by following the instructions in [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
        *   Learn about authoring command modules in [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
        *   Use Code Generation [Code Generation](https://github.com/Azure/aaz-dev-tools)
*   **Contribute Code:** Review the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) and adhere to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

## Data Collection and Telemetry

The Azure CLI collects telemetry data to improve its services.  You can disable telemetry by running `az config set core.collect_telemetry=no`.  See the [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) for more information.

## Reporting Issues and Feedback

*   Report bugs in the [Issues](https://github.com/Azure/azure-cli/issues) section of the GitHub repo.
*   Provide command-line feedback using the `az feedback` command.
*   \[Microsoft Internal] Contact the developer team via azpycli@microsoft.com.