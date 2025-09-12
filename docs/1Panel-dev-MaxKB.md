<p align="center"><img src= "https://github.com/1Panel-dev/maxkb/assets/52996290/c0694996-0eed-40d8-b369-322bf2a380bf" alt="MaxKB" width="300" /></p>

<h1 align="center">MaxKB: Build Powerful Enterprise-Grade AI Agents</h1>

<p align="center">Unlock the potential of your data with MaxKB, an open-source platform designed for building intelligent, enterprise-ready AI agents.</p>

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html#license-text"><img src="https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF" alt="License: GPL v3"></a>
  <a href="https://github.com/1Panel-dev/maxkb/releases/latest"><img src="https://img.shields.io/github/v/release/1Panel-dev/maxkb" alt="Latest release"></a>
  <a href="https://github.com/1Panel-dev/maxkb"><img src="https://img.shields.io/github/stars/1Panel-dev/maxkb?color=%231890FF&style=flat-square" alt="Stars"></a>
  <a href="https://hub.docker.com/r/1panel/maxkb"><img src="https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads" alt="Downloads"></a><br/>
  [<a href="/README_CN.md">中文(简体)</a>] | [<a href="/README.md">English</a>]
</p>

<hr/>

**MaxKB (Max Knowledge Brain)** is an open-source platform that empowers you to create advanced AI agents for enterprise applications. It combines Retrieval-Augmented Generation (RAG) pipelines with robust workflows and powerful tool-use capabilities.  Perfect for intelligent customer service, corporate knowledge bases, research, and education.  Find the original repository [here](https://github.com/1Panel-dev/MaxKB).

## Key Features

*   **RAG Pipeline:**
    *   Directly upload documents or automatically crawl online resources.
    *   Automatic text splitting and vectorization for enhanced accuracy.
    *   Reduce hallucinations in large language models (LLMs).
*   **Agentic Workflow:**
    *   A powerful workflow engine for orchestrating AI processes.
    *   Extensive function library and MCP tool-use support.
    *   Enable complex business scenario automation.
*   **Seamless Integration:**
    *   Rapid, zero-coding integration with third-party systems.
    *   Quickly add intelligent Q&A capabilities to existing platforms.
    *   Improve user satisfaction with enhanced interactions.
*   **Model Agnostic:**
    *   Supports a wide range of LLMs, including private and public models.
    *   Compatible with models like DeepSeek, Llama, Qwen, OpenAI, Claude, and Gemini.
*   **Multi-Modal Support:**
    *   Native support for text, image, audio, and video input and output.

## Quick Start (Docker)

Get started with MaxKB using Docker:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the MaxKB web interface at `http://your_server_ip:8080` using the following default credentials:

*   Username: `admin`
*   Password: `MaxKB@123..`

*For Chinese users facing Docker image pull issues, please refer to the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/).*

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

*   Frontend: [Vue.js](https://vuejs.org/)
*   Backend: [Python / Django](https://www.djangoproject.com/)
*   LLM Framework: [LangChain](https://www.langchain.com/)
*   Database: [PostgreSQL + pgvector](https://www.postgresql.org/)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=1Panel-dev/MaxKB&type=Date)](https://star-history.com/#1Panel-dev/MaxKB&Date)

## License

Licensed under The GNU General Public License version 3 (GPLv3).  You can find the license at:

<https://www.gnu.org/licenses/gpl-3.0.html>

---