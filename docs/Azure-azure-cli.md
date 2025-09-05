# Azure CLI: Manage Your Azure Resources with Ease

**Simplify cloud management with the Azure CLI, a powerful, cross-platform command-line interface for interacting with Microsoft Azure, enabling you to build, manage, and deploy Azure resources.** [Explore the Azure CLI on GitHub](https://github.com/Azure/azure-cli).

## Key Features:

*   **Cross-Platform Compatibility:** Works seamlessly across Windows, macOS, and Linux.
*   **Comprehensive Coverage:** Supports a wide array of Azure services.
*   **Intuitive Command Structure:** Uses a straightforward `az [group] [subgroup] [command] {parameters}` syntax.
*   **Tab Completion:** Enhance your productivity with tab completion for commands and parameters.
*   **Flexible Output:** Control output formatting with options like JSON, table, or TSV.
*   **Powerful Querying:** Use JMESPath queries for customized output (`--query` parameter).
*   **Scripting-Friendly:** Provides exit codes for scripting and automation.
*   **REST API Access:** Directly interact with Azure REST APIs using the `az rest` command.
*   **VS Code Integration:** Use the Azure CLI Tools extension for VS Code for enhanced command authoring and execution.

## Getting Started

*   **Installation:**  Follow the comprehensive [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for detailed installation instructions.
*   **Quick Start:** Get up and running quickly with the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).
*   **Command Help:** Use the `-h` parameter to get help for any command (e.g., `az storage -h`, `az vm create -h`).

## Highlights:

*   **Tab Completion:** Speed up command entry with tab completion.
*   **Querying:** Use the `--query` parameter and JMESPath to customize your output.
*   **Exit Codes:** Utilize exit codes for successful and unsuccessful operations.

## Common Scenarios & Effective Usage

*   **Output Formatting:** Customize output with JSON, table, or TSV formats.
*   **Command Chaining:** Pass values between commands for efficient workflows.
*   **Asynchronous Operations:** Manage async operations effectively.
*   **Resource Management:** Utilize the `az resource` commands for managing resources.
*   **REST API Integration:** Use `az rest` to interact with Azure REST APIs.
*   **Proxy Support:** Work seamlessly behind a proxy.
*   **Concurrent builds:** Configure concurrent builds as needed.
*   **[Tips for using Azure CLI effectively](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively)**

## Further Resources

*   **Samples and Snippets:** Explore numerous usage examples in the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) and the [Microsoft Learn documentation](https://learn.microsoft.com/cli/azure/overview).
*   **Azure CLI Tools for Visual Studio Code:** Enhance your workflow with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension.

## Developer Installation and Builds

*   **Docker:** Utilize preconfigured Docker images for the Azure CLI.  See [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list).
*   **Edge Builds:** Access the latest builds from the `dev` branch. See below for links to install:
    *   MSI: https://aka.ms/InstallAzureCliWindowsEdge
    *   Homebrew Formula: https://aka.ms/InstallAzureCliHomebrewEdge
    *   Ubuntu Bionic Deb: https://aka.ms/InstallAzureCliBionicEdge
    *   Ubuntu Focal Deb: https://aka.ms/InstallAzureCliFocalEdge
    *   Ubuntu Jammy Deb: https://aka.ms/InstallAzureCliJammyEdge
    *   RPM el8: https://aka.ms/InstallAzureCliRpmEl8Edge
*   **Builds from Specific Commits:** Get builds of arbitrary commits or pull requests. See: [Try new features before release](doc/try_new_features_before_release.md)
*   **Developer Setup:**
    *   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
    *   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
    *   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute

*   **Report Issues:**  Report bugs in the [Issues](https://github.com/Azure/azure-cli/issues) section.
*   **Provide Feedback:** Use the `az feedback` command to provide feedback.
*   **Contribute Code:** Follow the instructions in [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) and adhere to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
*   **Contact:** Contact the developer team via azpycli@microsoft.com for any further assistance.

## Data Collection & Telemetry

Telemetry collection is on by default. To opt out, run `az config set core.collect_telemetry=no`. For more details, review the [Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkID=824704).