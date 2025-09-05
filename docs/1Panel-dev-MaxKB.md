<p align="center"><img src= "https://github.com/1Panel-dev/maxkb/assets/52996290/c0694996-0eed-40d8-b369-322bf2a380bf" alt="MaxKB" width="300" /></p>

## MaxKB: Build Enterprise-Grade AI Agents with Ease

MaxKB is an open-source platform designed to empower you to build intelligent agents for a variety of enterprise applications.

<p align="center"><a href="https://trendshift.io/repositories/9113" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9113" alt="1Panel-dev%2FMaxKB | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a></p>

[![License: GPL v3](https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF)](https://www.gnu.org/licenses/gpl-3.0.html#license-text)
[![Latest Release](https://img.shields.io/github/v/release/1Panel-dev/maxkb)](https://github.com/1Panel-dev/maxkb/releases/latest)
[![Stars](https://img.shields.io/github/stars/1Panel-dev/maxkb?color=%231890FF&style=flat-square)](https://github.com/1Panel-dev/maxkb)
[![Docker Pulls](https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads)](https://hub.docker.com/r/1panel/maxkb)
[English](README.md) | [<a href="/README_CN.md">中文(简体)</a>]

### Key Features

*   **RAG Pipeline:** Easily ingest documents via upload or web crawling with automatic text splitting and vectorization, reducing LLM hallucinations.
*   **Agentic Workflow:** Powerful workflow engine with function library and MCP tool-use for orchestrating AI processes and handling complex business scenarios.
*   **Seamless Integration:** Quickly integrate with third-party systems, enabling intelligent Q&A capabilities without extensive coding.
*   **Model-Agnostic:** Supports a wide range of large language models (LLMs), including both private (DeepSeek, Llama, Qwen, etc.) and public (OpenAI, Claude, Gemini, etc.) models.
*   **Multi-Modal Support:** Native support for processing and generating text, images, audio, and video.

### Quick Start with Docker

Get up and running quickly using Docker:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the MaxKB web interface at `http://your_server_ip:8080` with the following default credentials:

*   username: `admin`
*   password: `MaxKB@123..`

**Note for Chinese Users:**  If you encounter Docker image pull issues, please refer to the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/) for alternative installation methods.

### Screenshots

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

### Technical Stack

*   Frontend: [Vue.js](https://vuejs.org/)
*   Backend: [Python / Django](https://www.djangoproject.com/)
*   LLM Framework: [LangChain](https://www.langchain.com/)
*   Database: [PostgreSQL + pgvector](https://www.postgresql.org/)

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=1Panel-dev/MaxKB&type=Date)](https://star-history.com/#1Panel-dev/MaxKB&Date)

### License

MaxKB is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html).

For more details, please visit the [original repository](https://github.com/1Panel-dev/MaxKB).