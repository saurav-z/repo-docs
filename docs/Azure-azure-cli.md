# Azure CLI: Manage Your Azure Resources with Ease

**The Azure CLI is a powerful, cross-platform command-line interface (CLI) designed to manage and administer Azure cloud resources effectively.** ([See the original repo](https://github.com/Azure/azure-cli))

## Key Features

*   **Cross-Platform:** Works seamlessly on Windows, macOS, and Linux.
*   **Comprehensive Command Set:** Provides commands for virtually all Azure services.
*   **Interactive & Scriptable:** Easily run commands interactively or automate tasks via scripts.
*   **Tab Completion:** Saves time and reduces errors with built-in tab completion.
*   **Flexible Output:** Supports various output formats (JSON, table, tsv) and custom queries using JMESPath.
*   **Azure Cloud Shell Integration:** Access the CLI directly from your browser via Azure Cloud Shell.
*   **VS Code Integration:** Enhance productivity with the Azure CLI Tools extension for Visual Studio Code, including IntelliSense, snippets, and more.

## Getting Started

### Installation

Detailed installation instructions are available in the [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli). Troubleshoot common installation issues via the [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md) guide.

### Basic Usage

Use the following format to execute commands:

```bash
$ az [group] [subgroup] [command] {parameters}
```

For help on specific commands, use the `-h` parameter:

```bash
$ az storage -h
$ az vm create -h
```

### Key Capabilities

*   **Tab Completion:** Quickly discover and select commands and parameters.
*   **Querying:** Use `--query` with JMESPath to customize and filter output.
*   **Exit Codes:** Understand command success or failure with clear exit codes.

### Effective Use and Common Scenarios

Explore these resources for maximizing your Azure CLI usage:

*   [Tips for using Azure CLI effectively](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively).
*   Output formatting (JSON, table, or TSV)
*   Passing values between commands
*   Asynchronous operations
*   Generic update arguments
*   Generic resource commands - `az resource`
*   REST API command - `az rest`
*   Quoting issues
*   Working behind a proxy
*   Concurrent builds

### Additional Resources

*   **Samples:** Explore numerous usage examples at the [GitHub samples repo](http://github.com/Azure/azure-cli-samples)
*   **Overview:** Discover more information at [https://learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview).

## Developer Installation

### Docker

Get preconfigured builds from the Docker image.

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

For the latest builds, try the edge builds.

| Package                 | Link                                      |
| :---------------------: | :---------------------------------------- |
|           MSI           | https://aka.ms/InstallAzureCliWindowsEdge |
|   Homebrew Formula    | https://aka.ms/InstallAzureCliHomebrewEdge |
| Ubuntu Bionic Deb     | https://aka.ms/InstallAzureCliBionicEdge  |
| Ubuntu Focal Deb      | https://aka.ms/InstallAzureCliFocalEdge   |
| Ubuntu Jammy Deb      | https://aka.ms/InstallAzureCliJammyEdge   |
|        RPM el8        | https://aka.ms/InstallAzureCliRpmEl8Edge  |

### Get builds of arbitrary commit or PR

[Try new features before release](doc/try_new_features_before_release.md)

## Developer Setup

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute

Contribute code by following the instructions provided in [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).

## Reporting Issues and Feedback

Report any issues in the [Issues](https://github.com/Azure/azure-cli/issues) section of the GitHub repo.

Provide feedback from the command line with the `az feedback` command.