# Azure CLI: Your Command-Line Interface for Azure Cloud Management

**Manage and automate your Azure cloud resources efficiently with the Azure CLI, a cross-platform command-line tool.**  Learn more and contribute on the [Azure CLI GitHub repository](https://github.com/Azure/azure-cli).

## Key Features

*   **Cross-Platform:** Works on Windows, macOS, and Linux.
*   **Comprehensive Azure Coverage:** Supports a wide range of Azure services.
*   **Intuitive Command Structure:** Uses a clear `az [group] [subgroup] [command] {parameters}` syntax.
*   **Tab Completion:**  Speeds up command entry with tab completion for groups, commands, and parameters.
*   **Powerful Querying:** Utilize the `--query` parameter and JMESPath for customized output.
*   **Scripting-Friendly:** Provides clear exit codes for automation and scripting.
*   **Output Formatting:**  Supports JSON, table, and TSV formats for flexible data presentation.
*   **Integration with VS Code:** Enhance your workflow with the Azure CLI Tools extension.

## Installation

Find detailed installation instructions for your platform:  [Install Guide](https://learn.microsoft.com/cli/azure/install-azure-cli)

Troubleshooting tips are available here: [Install Troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

## Getting Started

Learn how to effectively use the Azure CLI:  ["Get Started" Guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2)

*   **Example Usage:**
    ```bash
    $ az storage -h
    $ az vm create -h
    ```

## Highlights & Effective Usage

![Azure CLI Highlight Reel](doc/assets/AzBlogAnimation4.gif)

*   **Tab Completion:** Streamline your command entry.

    ```bash
    $ az vm show -g [tab][tab]
    ```

*   **Querying:** Customize your output with JMESPath.

    ```bash
    $ az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"
    ```

*   **Exit Codes:** Understand command success and failure.

    | Exit Code | Scenario                                                                  |
    | --------- | ------------------------------------------------------------------------- |
    | 0         | Command ran successfully.                                                |
    | 1         | Generic error; server returned bad status code, CLI validation failed, etc. |
    | 2         | Parser error; check input to command line.                                  |
    | 3         | Missing ARM resource; used for existence check from `show` commands.        |

*   **Common Scenarios:** Optimize your workflow.
    *   [Output formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
    *   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
    *   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
    *   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
    *   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
    *   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
    *   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
    *   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
    *   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

## More Resources

*   **Samples:**  Explore usage examples in the [GitHub samples repo](http://github.com/Azure/azure-cli-samples).
*   **Documentation:**  Find comprehensive information on [https://learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview).

## Azure CLI Tools for Visual Studio Code

Use the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension for enhanced productivity:

*   IntelliSense and code completion.
*   Command snippets.
*   Integrated terminal execution.
*   Side-by-side output display.
*   Documentation on hover.
*   Subscription and default display in status bar.

![Azure CLI Tools in Action](https://github.com/microsoft/vscode-azurecli/blob/main/images/in_action.gif?raw=true)

## Data Collection & Telemetry

The Azure CLI collects usage data to improve the tool.  You can opt out by running: `az config set core.collect_telemetry=no`.

[Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkID=824704).

## Reporting Issues and Providing Feedback

*   **Report bugs:** File issues in the [GitHub Issues](https://github.com/Azure/azure-cli/issues) section.
*   **Provide feedback:** Use the `az feedback` command.

## Developer Installation

### Docker

Use a preconfigured Docker image:

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

Access the latest development builds:

| Package             | Link                                       |
| :------------------ | :------------------------------------------- |
| MSI                 | https://aka.ms/InstallAzureCliWindowsEdge  |
| Homebrew Formula    | https://aka.ms/InstallAzureCliHomebrewEdge |
| Ubuntu Bionic Deb   | https://aka.ms/InstallAzureCliBionicEdge   |
| Ubuntu Focal Deb    | https://aka.ms/InstallAzureCliFocalEdge    |
| Ubuntu Jammy Deb    | https://aka.ms/InstallAzureCliJammyEdge    |
| RPM el8             | https://aka.ms/InstallAzureCliRpmEl8Edge   |

### Get builds of arbitrary commit or PR

See [Try new features before release](doc/try_new_features_before_release.md)

## Developer Setup

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contributing

Contribute to the Azure CLI project following the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

Find contribution guidelines in [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).