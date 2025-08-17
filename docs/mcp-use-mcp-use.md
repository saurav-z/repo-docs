<div align="center">
<div align="center" style="margin: 0 auto; max-width: 80%;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo-gh.jpg">
    <source media="(prefers-color-scheme: light)" srcset="static/logo-gh.jpg">
    <img alt="mcp use logo" src="./static/logo-gh.jpg" width="80%" style="margin: 20px auto;">
  </picture>
</div>

<div align="center">
  <h2>🎉 <strong>We're LIVE on Product Hunt!</strong> 🎉</h2>
  <p><strong>Support us today and help us reach #1!</strong></p>
  <a href="https://www.producthunt.com/products/mcp-use?embed=true&utm_source=badge-featured&utm_medium=badge&utm_source=badge-mcp&#0045;use" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1002629&theme=neutral&t=1754609432704" alt="mcp&#0045;use - Open&#0032;source&#0032;SDK&#0032;and&#0032;infra&#0032;for&#0032;MCP&#0032;servers&#0032;&#0038;&#0032;agents | Product Hunt" style="width: 220px; height: 54px;" width="250" height="54" /></a>
  <p>👆 <em>Click to upvote and leave a comment!</em></p>
</div>

<h1 align="center">🚀 **mcp-use: Unleash the Power of Open-Source MCP with Your LLM**</h1>
<p align="center">
    <a href="https://github.com/pietrozullo/mcp-use/stargazers" alt="GitHub stars">
        <img src="https://img.shields.io/github/stars/pietrozullo/mcp-use?style=social" /></a>
    <a href="https://pypi.org/project/mcp_use/" alt="PyPI Downloads">
        <img src="https://static.pepy.tech/badge/mcp-use" /></a>
    <a href="https://pypi.org/project/mcp_use/" alt="PyPI Version">
        <img src="https://img.shields.io/pypi/v/mcp_use.svg"/></a>
    <a href="https://github.com/mcp-use/mcp-use-ts" alt="TypeScript">
      <img src="https://img.shields.io/badge/TypeScript-mcp--use-3178C6?logo=typescript&logoColor=white" /></a>
    <a href="https://github.com/pietrozullo/mcp-use/blob/main/LICENSE" alt="License">
        <img src="https://img.shields.io/github/license/pietrozullo/mcp-use" /></a>
    <a href="https://docs.mcp-use.com" alt="Documentation">
        <img src="https://img.shields.io/badge/docs-mcp--use.com-blue" /></a>
    <a href="https://mcp-use.com" alt="Website">
        <img src="https://img.shields.io/badge/website-mcp--use.com-blue" /></a>
    </p>
    <p align="center">
    <a href="https://x.com/pietrozullo" alt="Twitter Follow - Pietro">
        <img src="https://img.shields.io/twitter/follow/Pietro?style=social" /></a>
    <a href="https://x.com/pederzh" alt="Twitter Follow - Luigi">
        <img src="https://img.shields.io/twitter/follow/Luigi?style=social" /></a>
    <a href="https://discord.gg/XkNkSkMz3V" alt="Discord">
        <img src="https://dcbadge.limes.pink/api/server/XkNkSkMz3V?style=flat" /></a>
</p>
</div>


[**mcp-use**](https://github.com/mcp-use/mcp-use) empowers developers to easily connect any Large Language Model (LLM) to various tools and build custom agents using the Model Context Protocol (MCP) without relying on closed-source solutions.

## Key Features

*   ✅ **LLM Compatibility**: Integrate with a wide range of LLMs supported by Langchain, including OpenAI, Anthropic, and more, provided they support tool calling.
*   🛠️ **Flexible Tool Access**: Connect your LLM to tools like web browsers, file operations, 3D Modeling and more.
*   🏗️ **Custom Agent Creation**: Build your own agents using any framework with the LangChain adapter or create new adapters.
*   🌐 **Multi-Server Support**: Utilize multiple MCP servers simultaneously within a single agent for enhanced functionality.
*   🔒 **Tool Access Control**: Implement tool restrictions to manage security and control agent capabilities.
*   ⚡ **Streaming Agent Output**: Supports asynchronous streaming of agent output for real-time feedback.
*   🖥️ **Sandboxed Execution**: Run MCP servers in a sandboxed environment using E2B's cloud infrastructure for secure and isolated execution.
*   ✨ **Direct Tool Calls**: Call MCP server tools directly without an LLM for programmatic control.
*   💻 **Code Builder:** Generate starter code with the interactive [code builder](https://mcp-use.com/builder).

## Quick Start

Install mcp-use:

```bash
pip install mcp-use
```

For detailed setup instructions, installation from source, and provider-specific instructions, refer to the detailed documentation.

## Examples

**Web Browsing with Playwright**

```python
# [Example code]
```

**Airbnb Search**

```python
# [Example code]
```

**Blender 3D Creation**

```python
# [Example code]
```

For additional use cases and configurations, please check out the full documentation.

## Configuration & Usage

mcp-use offers various configuration options, including HTTP connections, dynamic server selection, and server manager capabilities.

*   **HTTP Connection Example:**

    ```python
    # [Example code]
    ```

*   **Multi-Server Support** Configure multiple MCP servers and use them simultaneously.

*   **Dynamic Server Selection (Server Manager):** Enable the Server Manager to optimize agent tool selection.

*   **Tool Access Control:** Restrict which tools are available to the agent.

## Debugging

Enable debug mode for increased log verbosity to diagnose agent implementations.

### Debugging Modes

*   **Environment Variable:** Use the `DEBUG` environment variable.
*   **Programmatically:** Set the global debug flag in your code.
*   **Agent-Specific Verbosity:** Use the `verbose` parameter in `MCPAgent`.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=pietrozullo/mcp-use&type=Date)](https://www.star-history.com/#pietrozullo/mcp-use&Date)

## Contributing

Contributions are welcome! Review the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contributors

Thanks to all our amazing contributors!

<a href="https://github.com/mcp-use/mcp-use/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mcp-use/mcp-use" />
</a>

## Top Starred Dependents

<!-- gh-dependents-info-used-by-start -->

<table>
  <tr>
    <th width="400">Repository</th>
    <th>Stars</th>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/38653995?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/patchy631/ai-engineering-hub"><strong>patchy631/ai-engineering-hub</strong></a></td>
    <td>⭐ 15920</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/170207473?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/tavily-ai/meeting-prep-agent"><strong>tavily-ai/meeting-prep-agent</strong></a></td>
    <td>⭐ 129</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/164294848?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/buildfastwithai/gen-ai-experiments"><strong>buildfastwithai/gen-ai-experiments</strong></a></td>
    <td>⭐ 93</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/187057607?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/hud-evals/hud-python"><strong>hud-evals/hud-python</strong></a></td>
    <td>⭐ 76</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/20041231?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/krishnaik06/MCP-CRASH-Course"><strong>krishnaik06/MCP-CRASH-Course</strong></a></td>
    <td>⭐ 61</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/54944174?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/larksuite/lark-samples"><strong>larksuite/lark-samples</strong></a></td>
    <td>⭐ 35</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/892404?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/truemagic-coder/solana-agent-app"><strong>truemagic-coder/solana-agent-app</strong></a></td>
    <td>⭐ 30</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/8344498?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/schogini/techietalksai"><strong>schogini/techietalksai</strong></a></td>
    <td>⭐ 24</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/201161342?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/autometa-dev/whatsapp-mcp-voice-agent"><strong>autometa-dev/whatsapp-mcp-voice-agent</strong></a></td>
    <td>⭐ 22</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/100749943?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/Deniscartin/mcp-cli"><strong>Deniscartin/mcp-cli</strong></a></td>
    <td>⭐ 19</td>
  </tr>
</table>

<!-- gh-dependents-info-used-by-end -->

## Requirements

*   Python 3.11+
*   MCP implementation (like Playwright MCP)
*   LangChain and appropriate model libraries (OpenAI, Anthropic, etc.)

## License

MIT

## Citation

```bibtex
@software{mcp_use2025,
  author = {Zullo, Pietro},
  title = {MCP-Use: MCP Library for Python},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/pietrozullo/mcp-use}
}
```

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=732589b6-6850-4b8c-aa25-906c0979e426&page=README.md" />