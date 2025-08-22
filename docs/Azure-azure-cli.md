# Azure CLI: Your Command-Line Interface for Azure Cloud Management

**Simplify and automate your Azure cloud operations with the Azure CLI, a powerful, cross-platform command-line tool for managing Azure resources.** [(Original Repository)](https://github.com/Azure/azure-cli)

[![Python](https://img.shields.io/pypi/pyversions/azure-cli.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure-cli)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/cli/Azure.azure-cli?branchName=dev)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=246&branchName=dev)
[![Slack](https://img.shields.io/badge/Slack-azurecli.slack.com-blue.svg)](https://azurecli.slack.com)

The Azure CLI provides a rich set of commands to manage all aspects of Azure services, from virtual machines and storage to networking and databases. Automate tasks, streamline deployments, and manage your cloud infrastructure with ease.

## Key Features

*   **Cross-Platform Compatibility:** Run on Windows, macOS, and Linux.
*   **Comprehensive Coverage:** Manage a wide range of Azure services and resources.
*   **Scripting and Automation:** Integrate seamlessly into your scripts and automation workflows.
*   **Interactive Experience:** Utilize tab completion and intuitive command structures.
*   **Flexible Output:** Customize output formats (JSON, table, TSV) and query results with JMESPath.
*   **Integrated Tools:** Enhance your workflow with Visual Studio Code extensions.

## Installation

Follow these steps to install the Azure CLI:

*   Refer to the detailed [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli).
*   Troubleshooting information is available at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

## Usage

Use the following structure for running commands:

```bash
$ az [ group ] [ subgroup ] [ command ] {parameters}
```

### Get Started

*   Get in-depth instructions with the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).
*   Get help with the `-h` parameter (e.g., `$ az storage -h`, `$ az vm create -h`).

### Highlights

*   **Tab Completion:** Enhance productivity with tab completion for groups, commands, and parameters.
*   **Querying:** Use the `--query` parameter and JMESPath to customize your output.
*   **Exit Codes:** Utilize exit codes for scripting and automation.

## Common Scenarios

Learn how to use the Azure CLI effectively by exploring these common scenarios:

*   [Output formatting (JSON, table, or TSV)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

## More Resources

*   Explore more samples and snippets in the [GitHub samples repo](http://github.com/Azure/azure-cli-samples).
*   Refer to the [Microsoft Azure CLI Overview](https://learn.microsoft.com/cli/azure/overview) for detailed usage and examples.

## Visual Studio Code Integration

Enhance your development workflow with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension for Visual Studio Code:

*   IntelliSense for commands and their arguments.
*   Snippets for commands.
*   Run commands in the integrated terminal or a side-by-side editor.
*   Documentation on mouse hover.
*   Subscription and defaults display in the status bar.

## Data Collection & Telemetry

This tool collects telemetry data.

*   Disable telemetry with: `az config set core.collect_telemetry=no`.
*   See our privacy statement: https://go.microsoft.com/fwlink/?LinkID=824704.

## Reporting Issues and Feedback

*   Report bugs via the [Issues](https://github.com/Azure/azure-cli/issues) section.
*   Provide feedback from the command line: `az feedback`.

## Developer Installation

### Docker

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

Get the latest build from the `dev` branch using these links:

|      Package      | Link                                       |
|:-----------------:|:-------------------------------------------|
|        MSI        | https://aka.ms/InstallAzureCliWindowsEdge  |
| Homebrew Formula  | https://aka.ms/InstallAzureCliHomebrewEdge |
| Ubuntu Bionic Deb | https://aka.ms/InstallAzureCliBionicEdge   |
| Ubuntu Focal Deb  | https://aka.ms/InstallAzureCliFocalEdge    |
| Ubuntu Jammy Deb  | https://aka.ms/InstallAzureCliJammyEdge    |
|      RPM el8      | https://aka.ms/InstallAzureCliRpmEl8Edge   |

### Get builds of arbitrary commit or PR

See: [Try new features before release](doc/try_new_features_before_release.md)

## Developer Setup

For development setup and contributions, see:

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute Code

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com).
To contribute, follow the instructions in [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).