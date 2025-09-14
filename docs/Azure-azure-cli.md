# Azure CLI: Your Command-Line Interface for Effortless Azure Management

**Manage your Azure resources with ease using the Azure CLI, a powerful and versatile command-line tool.**  [Check out the original repository](https://github.com/Azure/azure-cli).

**Key Features:**

*   **Multi-Platform Compatibility:** Run the Azure CLI on Windows, macOS, and Linux.
*   **Intuitive Command Structure:**  Use the `az` command followed by resource groups, subgroups, and commands for clear and concise operation.
*   **Interactive Help:** Get help with commands and parameters using the `-h` flag (e.g., `az vm create -h`).
*   **Tab Completion:** Improve efficiency with tab completion for groups, commands, and parameters.
*   **Flexible Output Formatting:** Customize your output using `--query` and JMESPath for tailored results.  Supports JSON, table, and TSV formats.
*   **Exit Codes for Scripting:** Use exit codes for easy scripting and automation.
*   **VS Code Integration:**  Leverage the Azure CLI Tools extension in Visual Studio Code for IntelliSense, snippets, and in-editor command execution.

## Getting Started

**Installation:** Refer to the [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for detailed instructions. Troubleshooting tips are available at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

**Basic Usage:**  Use the following format for commands:

```bash
$ az [ group ] [ subgroup ] [ command ] {parameters}
```

**Example:**

```bash
$ az storage account create --name myaccount --resource-group myresourcegroup --location westus
```

**Learn More:** Explore the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2) for in-depth instructions.

## Advanced Usage and Tips

*   **Output Formatting:**  Control output formats (JSON, table, tsv).
*   **Command Chaining:** Pass values between commands.
*   **Asynchronous Operations:** Manage long-running tasks.
*   **Generic Update Arguments:** Use update arguments for resource management.
*   **Generic Resource Commands:** Leverage `az resource` for broad resource control.
*   **REST API Command:** Utilize `az rest` for direct API interaction.
*   **Work Behind a Proxy:** Configure the CLI for proxy environments.
*   **Concurrent Builds:** Optimize builds for efficiency.

## Additional Resources

*   **Samples and Snippets:** [GitHub samples repo](http://github.com/Azure/azure-cli-samples) and [Microsoft Learn](https://learn.microsoft.com/cli/azure/overview).
*   **Visual Studio Code Extension:**  Install [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) for enhanced development.

## Developer Information

**Telemetry:**  Telemetry collection is enabled by default.  You can disable it with `az config set core.collect_telemetry=no`.

**Reporting Issues:**  Report bugs via the [GitHub Issues](https://github.com/Azure/azure-cli/issues) section.

**Feedback:**  Provide feedback from the command line with the `az feedback` command.

**Developer Setup:** Instructions for setting up a development environment: [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md).

**Contribute:** [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).