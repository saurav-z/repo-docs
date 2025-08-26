# Azure CLI: Manage Your Azure Resources from the Command Line

**The Azure CLI is a powerful, cross-platform command-line interface (CLI) for managing Azure resources.**

[Go to the Azure CLI GitHub Repository](https://github.com/Azure/azure-cli)

## Key Features

*   **Cross-Platform:** Works on Windows, macOS, and Linux.
*   **Comprehensive:** Provides commands for managing a vast range of Azure services.
*   **Easy to Use:**  Simple syntax, tab completion, and interactive help make it easy to learn and use.
*   **Flexible Output:** Supports JSON, table, and TSV output formats, plus custom output with JMESPath queries.
*   **Scripting Friendly:**  Returns exit codes for scripting and automation.
*   **VS Code Integration:**  The Azure CLI Tools extension for Visual Studio Code provides IntelliSense, snippets, and in-editor command execution.

## Installation

Install the Azure CLI on your operating system by following the detailed instructions in the [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli).

**Troubleshooting:** Refer to the [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md) guide for common issues.

### Developer Installation

*   **Docker:** Access a preconfigured Azure CLI environment using Docker.
*   **Edge Builds:** Get the latest builds from the `dev` branch for testing new features.
    *   [MSI](https://aka.ms/InstallAzureCliWindowsEdge)
    *   [Homebrew Formula](https://aka.ms/InstallAzureCliHomebrewEdge)
    *   [Ubuntu Bionic Deb](https://aka.ms/InstallAzureCliBionicEdge)
    *   [Ubuntu Focal Deb](https://aka.ms/InstallAzureCliFocalEdge)
    *   [Ubuntu Jammy Deb](https://aka.ms/InstallAzureCliJammyEdge)
    *   [RPM el8](https://aka.ms/InstallAzureCliRpmEl8Edge)
*   **Developer Setup:** Set up a development environment to contribute to the CLI.

## Usage

Use the following command structure:

```bash
$ az [ group ] [ subgroup ] [ command ] {parameters}
```

### Getting Started

*   Refer to the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2) for detailed instructions.
*   Get help with the `-h` parameter:  `az storage -h` or `az vm create -h`

### Highlights

*   **Tab Completion:** Speed up your workflow with tab completion for groups, commands, and parameters.
*   **Querying:** Use the `--query` parameter and JMESPath syntax to customize output.
*   **Exit Codes:** Use exit codes in scripts for different scenarios.

### Common Scenarios and Effective Use

Learn how to use the Azure CLI effectively with these common scenarios:

*   [Output formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

### Additional Resources

*   **Samples:** Explore more usage examples in our [GitHub samples repo](http://github.com/Azure/azure-cli-samples) or [https://learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview).
*   **VS Code Extension:**  Enhance your experience with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension in Visual Studio Code.

## Data Collection

The software may collect information about you and your use of the software and send it to Microsoft. You may turn off the telemetry as described in the repository.

### Telemetry Configuration

Telemetry is enabled by default. Opt-out by running `az config set core.collect_telemetry=no`.

## Reporting Issues and Providing Feedback

*   **Report Bugs:** File issues in the [Issues](https://github.com/Azure/azure-cli/issues) section of our GitHub repo.
*   **Provide Feedback:** Use the `az feedback` command from the command line.
*   **Contact the Development Team:** \[Microsoft internal] azpycli@microsoft.com.

## Contributing

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

If you would like to become an active contributor to this project please
follow the instructions provided in [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).