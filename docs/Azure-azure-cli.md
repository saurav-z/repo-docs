# Azure CLI: Manage Azure Resources from the Command Line

**The Azure CLI is your cross-platform command-line interface for managing and interacting with Azure services, providing a powerful and flexible way to control your cloud infrastructure.** [Learn more on GitHub](https://github.com/Azure/azure-cli).

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.
*   **Comprehensive Azure Support:** Manage a wide range of Azure services.
*   **Intuitive Command Structure:** Uses a clear `az [group] [subgroup] [command] {parameters}` syntax.
*   **Tab Completion:** Speeds up command entry with intelligent suggestions.
*   **JMESPath Querying:** Customize output with flexible query capabilities.
*   **Scripting Friendly:** Provides specific exit codes for automation.
*   **VS Code Integration:** Includes an extension for enhanced development, including IntelliSense, snippets, and terminal integration.
*   **Output Formatting:**  Supports `json`, `table`, and `tsv` for easy data consumption.
*   **REST API Access:** Directly call REST APIs using the `az rest` command.

## Installation

Detailed installation instructions are available in the [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli).

*   Troubleshooting guide: [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

## Usage

```bash
$ az [group] [subgroup] [command] {parameters}
```

### Getting Started

Refer to the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2) for comprehensive instructions.

*   **Help:** Get help for any command using `-h`:

    ```bash
    $ az storage -h
    $ az vm create -h
    ```

### Common Scenarios and Effective Use

*   **Output Formatting:**  Choose your preferred output format (JSON, table, or TSV).
*   **Command Chaining:** Pass values between commands.
*   **Asynchronous Operations:** Manage long-running tasks.
*   **Generic Updates:**  Use generic update arguments for efficient resource modifications.
*   **Resource Management:** Leverage generic resource commands (`az resource`).
*   **REST API Access:** Use the `az rest` command to interact directly with the Azure REST API.
*   **Proxy Support:** Work behind a proxy server.
*   **Concurrent Builds:** Optimizing concurrent build processes.

For more usage examples and tips, see: [Tips for using Azure CLI effectively](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively).

## Examples

```bash
# List virtual machines with specific details
az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"
```

## Contributing and Feedback

*   **Report Issues:**  File bugs and feature requests in the [GitHub Issues](https://github.com/Azure/azure-cli/issues) section.
*   **Provide Feedback:** Use the `az feedback` command from the command line.
*   **Contribute Code:** See the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) for contribution instructions.

## Developer Installation

*   **Docker:** Use a preconfigured Docker image. See [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list).
*   **Edge Builds:** Access the latest builds from the `dev` branch. Links for various operating systems and installation methods are provided.

### Developer Setup

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Data Collection

The Azure CLI collects usage data to improve the service.  You can opt-out of telemetry collection by running `az config set core.collect_telemetry=no`. More information can be found in the privacy statement: [https://go.microsoft.com/fwlink/?LinkID=824704](https://go.microsoft.com/fwlink/?LinkID=824704).