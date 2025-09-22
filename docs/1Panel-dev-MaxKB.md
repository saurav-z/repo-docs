<p align="center"><img src= "https://github.com/1Panel-dev/maxkb/assets/52996290/c0694996290" alt="MaxKB" width="300" /></p>

<h1 align="center">MaxKB: The Open-Source Platform for Enterprise-Grade AI Agents</h1>

MaxKB empowers you to build intelligent agents with advanced Retrieval-Augmented Generation (RAG) and workflow capabilities.

<p align="center">
  <a href="https://github.com/1Panel-dev/MaxKB" target="_blank">
    <img src="https://img.shields.io/github/stars/1Panel-dev/MaxKB?style=flat-square&color=%231890FF" alt="Stars">
  </a>
  <a href="https://github.com/1Panel-dev/maxkb/releases/latest">
    <img src="https://img.shields.io/github/v/release/1Panel-dev/maxkb" alt="Latest Release">
  </a>
  <a href="https://hub.docker.com/r/1panel/maxkb">
    <img src="https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads" alt="Docker Pulls">
  </a>
  <a href="https://www.gnu.org/licenses/gpl-3.0.html#license-text">
    <img src="https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF" alt="License: GPL v3">
  </a>
</p>

[English](README.md) | [中文(简体)](/README_CN.md)

## Key Features

*   **Advanced RAG Pipelines:** Easily ingest documents through direct upload or web crawling. Automated text splitting and vectorization for enhanced accuracy and reduced hallucinations.
*   **Agentic Workflow Engine:** Orchestrate complex AI processes with a powerful workflow engine, function libraries, and MCP tool-use.
*   **Seamless Integration:** Integrate MaxKB into your existing systems quickly and easily with zero coding, enhancing user satisfaction.
*   **Model-Agnostic:** Supports a wide range of large language models (LLMs), including private models (DeepSeek, Llama, Qwen) and public models (OpenAI, Claude, Gemini).
*   **Multi-Modal Support:** Native support for text, image, audio, and video input and output.

## Quick Start with Docker

Get started with MaxKB in minutes using Docker:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the web interface at `http://your_server_ip:8080` using the default credentials:

*   **Username:** `admin`
*   **Password:** `MaxKB@123..`

**Note for Chinese users:**  If you encounter Docker image pull issues, please refer to the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/) for installation instructions.

## Screenshots

<table style="border-collapse: collapse; border: 1px solid black;">
  <tr>
    <td style="padding: 5px;background-color:#fff;"><img src= "https://github.com/user-attachments/assets/eb285512-a66a-4752-8941-c65ed1592238" alt="MaxKB Demo1"   /></td>
    <td style="padding: 5px;background-color:#fff;"><img src= "https://github.com/user-attachments/assets/f732f1f5-472c-4fd2-93c1-a277eda83d04" alt="MaxKB Demo2"   /></td>
  </tr>
  <tr>
    <td style="padding: 5px;background-color:#fff;"><img src= "https://github.com/user-attachments/assets/c927474a-9a23-4830-822f-5db26025c9b2" alt="MaxKB Demo3"   /></td>
    <td style="padding: 5px;background-color:#fff;"><img src= "https://github.com/user-attachments/assets/e6268996-a46d-4e58-9f30-31139df78ad2" alt="MaxKB Demo4"   /></td>
  </tr>
</table>

## Technical Stack

*   **Frontend:** [Vue.js](https://vuejs.org/)
*   **Backend:** [Python / Django](https://www.djangoproject.com/)
*   **LLM Framework:** [LangChain](https://www.langchain.com/)
*   **Database:** [PostgreSQL + pgvector](https://www.postgresql.org/)

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=1Panel-dev/MaxKB&type=Date)](https://star-history.com/#1Panel-dev/MaxKB&Date)

## License

MaxKB is licensed under the [GNU General Public License v3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html).  See the LICENSE file for more details.

**[View the MaxKB Repository on GitHub](https://github.com/1Panel-dev/MaxKB)**
```

Key improvements and SEO considerations:

*   **Clear, Concise Title:**  Using "MaxKB: The Open-Source Platform for Enterprise-Grade AI Agents" clearly identifies the project and includes relevant keywords.
*   **One-Sentence Hook:**  The opening sentence immediately conveys the value proposition.
*   **Keyword Optimization:**  Includes keywords like "open-source," "AI agents," "RAG," "enterprise-grade," and LLM to improve search visibility.
*   **Bulleted Key Features:**  Uses concise bullet points to highlight the most important benefits.
*   **Actionable Quick Start:**  Provides a clear, copy-and-paste Docker command.
*   **Formatted Headings:** Uses headings for better readability and SEO structure.
*   **Links:** Includes a direct link back to the original GitHub repository at the end of the document.  Also links to the relevant documentation and tech stack components.
*   **Concise Language:** Streamlined text for better readability and impact.
*   **Updated Badges:** While the original badges were kept, a few key ones were added to increase prominence.
*   **Removed redundant phrases and information.**