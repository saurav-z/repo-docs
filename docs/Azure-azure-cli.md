# Azure CLI: Command-Line Interface for Microsoft Azure

**Manage and automate your Azure resources with the powerful and versatile Azure CLI, a cross-platform command-line tool.**  [Explore the Azure CLI Repo](https://github.com/Azure/azure-cli)

## Key Features

*   **Cross-Platform Compatibility:** Use the Azure CLI on Windows, macOS, and Linux.
*   **Comprehensive Resource Management:** Manage a wide array of Azure services, from virtual machines to storage and networking.
*   **Interactive and Scriptable:** Execute commands interactively or automate tasks through scripting.
*   **Powerful Querying with JMESPath:**  Extract and format data with the `--query` parameter and JMESPath syntax.
*   **Tab Completion:**  Improve productivity with tab completion for commands and parameters.
*   **Customizable Output:** Choose your preferred output format (JSON, table, TSV) or customize using the `--query` parameter.
*   **Exit Codes for Scripting:**  Understand command execution results through standardized exit codes.
*   **Integration with VS Code:** Enhance your workflow with the Azure CLI Tools extension.
*   **Regular Updates:** Stay up-to-date with the latest features and improvements.

## Getting Started

*   **Installation:**  Follow the [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for detailed instructions.  Troubleshooting tips are available at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).
*   **Quick Start:** Get started quickly with the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).
*   **Command Structure:** Utilize the `az [group] [subgroup] [command] {parameters}` format.
*   **Help:** Get help and usage information with the `-h` parameter (e.g., `az storage -h`).

## Core Concepts & Scenarios

*   **Output Formatting:**  Control output with JSON, table, or TSV formats.
*   **Command Chaining:** Pass values between commands for streamlined workflows.
*   **Async Operations:** Manage asynchronous tasks effectively.
*   **Generic Update Arguments:** Leverage common update arguments for easy modification.
*   **Generic Resource Commands:** Use `az resource` for managing Azure resources.
*   **REST API Command:** Use `az rest` to interact directly with Azure REST APIs.
*   **Work Behind a Proxy:** Configure the CLI for use with proxies.
*   **Concurrent Builds:** Learn how to handle concurrent builds effectively.
*   **Output formatting (json, table, or tsv)** [Output formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   **Pass values from one command to another** [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   **Async operations** [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   **Generic update arguments** [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   **Generic resource commands - `az resource`** [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   **REST API command - `az rest`** [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   **Quoting issues** [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   **Work behind a proxy** [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   **Concurrent builds** [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

## Developer Installation

*   **Docker:** Preconfigured Docker images are available.  See [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list) for available versions.
*   **Edge Builds:** Get the latest builds from the `dev` branch:
    *   MSI: https://aka.ms/InstallAzureCliWindowsEdge
    *   Homebrew Formula: https://aka.ms/InstallAzureCliHomebrewEdge
    *   Ubuntu Bionic Deb: https://aka.ms/InstallAzureCliBionicEdge
    *   Ubuntu Focal Deb: https://aka.ms/InstallAzureCliFocalEdge
    *   Ubuntu Jammy Deb: https://aka.ms/InstallAzureCliJammyEdge
    *   RPM el8: https://aka.ms/InstallAzureCliRpmEl8Edge
*   **Arbitrary Builds:** Get builds of a specific commit or pull request by following the instructions at: [Try new features before release](doc/try_new_features_before_release.md)
*   **Developer Setup:** Set up a development environment by following the instructions at:
    *   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
    *   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
    *   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute

This project welcomes contributions! See the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/) and [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) for details.

## Support & Feedback

*   **Report Issues:** File issues in the [Issues](https://github.com/Azure/azure-cli/issues) section.
*   **Provide Feedback:** Use the `az feedback` command from the command line.