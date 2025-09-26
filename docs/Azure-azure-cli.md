# Azure CLI: Manage Your Azure Resources with Ease

**The Azure CLI is a powerful cross-platform command-line tool that lets you manage and configure your Azure resources from a shell or terminal.**  [Learn more about the Azure CLI on the official GitHub repo](https://github.com/Azure/azure-cli).

[![Python](https://img.shields.io/pypi/pyversions/azure-cli.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure-cli)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/cli/Azure.azure-cli?branchName=dev)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=246&branchName=dev)
[![Slack](https://img.shields.io/badge/Slack-azurecli.slack.com-blue.svg)](https://azurecli.slack.com)

## Key Features

*   **Cross-Platform:** Use the Azure CLI on Windows, macOS, and Linux.
*   **Cloud Shell Integration:** Run the CLI directly from your browser using [Azure Cloud Shell](https://portal.azure.com/#cloudshell).
*   **Tab Completion:**  Efficiently navigate commands with tab completion for groups, commands, and parameters.
*   **Flexible Output:** Customize output using `--query` and JMESPath for precise data retrieval.
*   **Scripting Support:** Leverage exit codes for seamless integration into your scripts.
*   **Extensive Documentation:** Comprehensive guides and examples available on [Microsoft Learn](https://learn.microsoft.com/cli/azure/overview).
*   **VS Code Integration:** Enhance your workflow with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension for VS Code, offering IntelliSense, snippets, and more.

## Installation

Follow the steps in the [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for detailed instructions for your specific operating system. Troubleshoot common issues using the [installation troubleshooting guide](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

## Usage

The general command syntax is:

```bash
az [ group ] [ subgroup ] [ command ] {parameters}
```

### Getting Started

For comprehensive guidance, consult the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2). You can access help documentation for any command using the `-h` parameter:

```bash
az storage -h
az vm create -h
```

## Highlights

The Azure CLI offers several features to enhance your workflow:

![Azure CLI Highlight Reel](doc/assets/AzBlogAnimation4.gif)

*   **Tab Completion:** Speed up your command entry.
*   **Query:** Refine your output using the `--query` parameter and JMESPath.
    ```bash
    az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"
    ```
*   **Exit Codes:**  Utilize exit codes for efficient scripting.

    | Exit Code | Scenario                                                        |
    | --------- | --------------------------------------------------------------- |
    | 0         | Command ran successfully.                                        |
    | 1         | Generic error; server returned bad status code, CLI validation failed, etc. |
    | 2         | Parser error; check input to command line.                      |
    | 3         | Missing ARM resource; used for existence check from `show` commands.    |

## Common Scenarios

Explore these common scenarios to use the Azure CLI effectively:

*   [Output formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

## More Samples and Snippets

Find more examples and usage scenarios:

*   [GitHub samples repo](http://github.com/Azure/azure-cli-samples)
*   [Microsoft Learn overview](https://learn.microsoft.com/cli/azure/overview)

## Developer Installation

### Docker

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

Obtain the latest builds from the `dev` branch using these links:

| Package             | Link                                        |
| ------------------- | ------------------------------------------- |
| MSI                 | https://aka.ms/InstallAzureCliWindowsEdge   |
| Homebrew Formula    | https://aka.ms/InstallAzureCliHomebrewEdge  |
| Ubuntu Bionic Deb   | https://aka.ms/InstallAzureCliBionicEdge    |
| Ubuntu Focal Deb    | https://aka.ms/InstallAzureCliFocalEdge     |
| Ubuntu Jammy Deb    | https://aka.ms/InstallAzureCliJammyEdge     |
| RPM el8             | https://aka.ms/InstallAzureCliRpmEl8Edge    |

Examples for installation are provided.

### Get builds of arbitrary commit or PR

[Try new features before release](doc/try_new_features_before_release.md)

## Developer Setup

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute Code

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information, refer to the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com).
For contribution guidelines see [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).

## Reporting Issues and Feedback

*   File issues in the [Issues](https://github.com/Azure/azure-cli/issues) section of the GitHub repo.
*   Provide feedback from the command line using the `az feedback` command.
*   \[Microsoft internal] Contact the developer team via azpycli@microsoft.com.

## Data Collection and Telemetry

The Azure CLI collects telemetry data to improve the product. You can opt-out by running:

```bash
az config set core.collect_telemetry=no
```

For more information on data collection, refer to the privacy statement at https://go.microsoft.com/fwlink/?LinkID=824704.