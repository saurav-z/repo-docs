# Azure CLI: Command-Line Interface for Managing Azure Resources

**Simplify your Azure cloud management with the Azure CLI, a powerful, cross-platform command-line tool.** [Learn more about the Azure CLI](https://github.com/Azure/azure-cli).

## Key Features

*   **Cross-Platform:** Compatible with Windows, macOS, and Linux.
*   **Comprehensive Azure Coverage:** Manage a wide range of Azure services.
*   **Interactive and Scriptable:** Execute commands interactively or automate tasks with scripts.
*   **Tab Completion:** Quickly find commands and parameters with built-in tab completion.
*   **Flexible Output Formatting:** Customize output with JSON, table, or TSV formats.
*   **Powerful Querying:** Use JMESPath for advanced output filtering and transformation.
*   **Extensive Documentation:** Access detailed help and examples directly from the CLI.
*   **VS Code Integration:** Enhance your workflow with the Azure CLI Tools extension for Visual Studio Code.

## Getting Started

### Installation

Detailed installation instructions are available in the [Azure CLI installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli).  Troubleshooting tips can be found at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

### Usage

Use the following syntax:

```bash
$ az [group] [subgroup] [command] {parameters}
```

Get started quickly with the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).
Find help using the `-h` parameter:

```bash
$ az storage -h
$ az vm create -h
```

##  Highlights & Advanced Features

*   **Tab Completion:** Streamline command input with tab completion for groups, commands, and parameters.
*   **Querying with JMESPath:** Customize your output with the `--query` parameter and [JMESPath](http://jmespath.org/) query syntax.
*   **Exit Codes:** Use exit codes for scripting and automation purposes.
*   **Output Formatting:** Change your default output format using the `az configure` command.
*   **Common scenarios and use Azure CLI effectively**: Check [Tips for using Azure CLI effectively](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively).
*   **More samples and snippets**: Check [GitHub samples repo](http://github.com/Azure/azure-cli-samples) or [https://learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview).

##  Developer Installation

### Docker

Use preconfigured Azure CLI Docker images.

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

Access the latest builds from the `dev` branch.

#### Installation Links:

|      Package      | Link                                       |
|:-----------------:|:-------------------------------------------|
|        MSI        | https://aka.ms/InstallAzureCliWindowsEdge  |
| Homebrew Formula  | https://aka.ms/InstallAzureCliHomebrewEdge |
| Ubuntu Bionic Deb | https://aka.ms/InstallAzureCliBionicEdge   |
| Ubuntu Focal Deb  | https://aka.ms/InstallAzureCliFocalEdge    |
| Ubuntu Jammy Deb  | https://aka.ms/InstallAzureCliJammyEdge    |
|      RPM el8      | https://aka.ms/InstallAzureCliRpmEl8Edge   |

#### Examples:

*   Homebrew:
    ```bash
    curl --location --silent --output azure-cli.rb https://aka.ms/InstallAzureCliHomebrewEdge
    brew install --build-from-source azure-cli.rb
    ```
*   Ubuntu Jammy:
    ```bash
    curl --location --silent --output azure-cli_jammy.deb https://aka.ms/InstallAzureCliJammyEdge && dpkg -i azure-cli_jammy.deb
    ```
*   RHEL 8 or CentOS Stream 8:
    ```bash
    dnf install -y $(curl --location --silent --output /dev/null --write-out %{url_effective} https://aka.ms/InstallAzureCliRpmEl8Edge)
    ```

### Get builds of arbitrary commit or PR

If you would like to get builds of arbitrary commit or PR, see:

[Try new features before release](doc/try_new_features_before_release.md)

### Developer setup

Configure your development environment:

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contributing

This project welcomes contributions. See the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) and the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/).

## Reporting Issues and Feedback

Report bugs in the [Issues](https://github.com/Azure/azure-cli/issues) section. Provide feedback using the `az feedback` command.