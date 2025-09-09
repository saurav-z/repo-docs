# Azure CLI: Manage Your Azure Resources from the Command Line

**Simplify Azure resource management with the Azure CLI, a powerful, cross-platform command-line tool for interacting with your Azure cloud environment.**  [Learn More](https://github.com/Azure/azure-cli)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.
*   **Comprehensive Azure Coverage:** Supports a vast array of Azure services and resources.
*   **Intuitive Command Structure:** Uses a clear `az [group] [subgroup] [command] {parameters}` syntax.
*   **Tab Completion:** Improves efficiency with intelligent tab completion for commands and parameters.
*   **Flexible Output Formatting:** Offers JSON, table, and TSV output formats, customizable via the `--query` parameter and JMESPath.
*   **Scripting Support:** Provides exit codes for easy integration into scripts and automation workflows.
*   **Extensible:** Integrates with Visual Studio Code for enhanced development with IntelliSense, snippets, and more.
*   **REST API Command:** `az rest` allows you to call Azure REST APIs directly.

## Installation

Refer to the detailed [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for your operating system.

*   **Troubleshooting:** Find solutions to common installation issues at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

## Getting Started

Get started quickly with the Azure CLI using the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).

**Example Usage:**

```bash
$ az storage -h  # Get help for storage commands
$ az vm create -h # Get help for creating virtual machines
```

## Highlights

*   **Tab Completion:** Enhance your workflow with tab completion for commands and parameters.
*   **Query:** Use the `--query` parameter with [JMESPath](http://jmespath.org/) to customize output.
*   **Exit Codes:** Utilize exit codes for scripting and automation:

    *   `0`: Command successful.
    *   `1`: Generic error.
    *   `2`: Parser error.
    *   `3`: Missing ARM resource.

## Common Scenarios & Effective Usage

Explore these resources for efficient Azure CLI usage:

*   [Output formatting](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values between commands](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

## Visual Studio Code Integration

Enhance your development experience with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension:

*   IntelliSense for commands.
*   Command snippets.
*   Integrated terminal execution.
*   Documentation on hover.

## Data Collection & Telemetry

The Azure CLI collects usage data to improve the tool. You can opt out using the command `az config set core.collect_telemetry=no`. Learn more in the [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704).

## Reporting Issues and Feedback

*   **Report issues:** File issues in the [GitHub repository](https://github.com/Azure/azure-cli/issues).
*   **Provide feedback:** Use the `az feedback` command.
*   **Contact the team:**  [azpycli@microsoft.com](mailto:azpycli@microsoft.com) (Microsoft internal).

## Developer Installation

### Docker

Use a preconfigured Docker image:

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

Get the latest builds from the `dev` branch:

*   **MSI:** https://aka.ms/InstallAzureCliWindowsEdge
*   **Homebrew:** https://aka.ms/InstallAzureCliHomebrewEdge
*   **Debian/RPM:** Various links provided for Ubuntu, RHEL, and CentOS.
*   **Pip (Virtual Environment):**
    ```bash
    $ python3 -m venv env
    $ . env/bin/activate
    $ pip3 install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge --upgrade-strategy=eager
    ```

### Get Builds of Arbitrary Commit or PR

*   [Try new features before release](doc/try_new_features_before_release.md)

## Developer Setup

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  See the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com).

Contribute code by following the instructions provided in [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).