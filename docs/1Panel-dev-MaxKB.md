<p align="center">
  <img src="https://github.com/1Panel-dev/maxkb/assets/52996290/c0694996-0eed-40d8-b369-322bf2a380bf" alt="MaxKB" width="300" />
</p>

<h1 align="center">MaxKB: Build Powerful Enterprise AI Agents with Ease</h1>

MaxKB is an open-source platform empowering developers to create intelligent agents for various business applications, offering robust features for RAG pipelines, agentic workflows, and seamless integration.  Learn more and contribute on [the original GitHub repository](https://github.com/1Panel-dev/MaxKB).

<p align="center">
  <a href="https://trendshift.io/repositories/9113" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/9113" alt="1Panel-dev%2FMaxKB | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</p>

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html#license-text"><img src="https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF" alt="License: GPL v3"></a>
  <a href="https://github.com/1Panel-dev/maxkb/releases/latest"><img src="https://img.shields.io/github/v/release/1Panel-dev/maxkb" alt="Latest release"></a>
  <a href="https://github.com/1Panel-dev/maxkb"><img src="https://img.shields.io/github/stars/1Panel-dev/maxkb?color=%231890FF&style=flat-square" alt="Stars"></a>
  <a href="https://hub.docker.com/r/1panel/maxkb"><img src="https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads" alt="Download"></a>
  <br/>
  [<a href="/README_CN.md">中文(简体)</a>] | [<a href="/README.md">English</a>]
</p>

<hr/>

## Key Features

*   **RAG Pipeline for Enhanced Accuracy:**  Upload documents, automatically split and vectorize text, and reduce hallucinations for superior question-answering experiences.
*   **Agentic Workflow Engine:** Utilize a powerful workflow engine, function library, and MCP tool-use for orchestrating AI processes in complex scenarios.
*   **Seamless Integration:** Quickly integrate MaxKB into existing third-party systems with minimal coding, adding intelligent Q&A capabilities.
*   **Model-Agnostic Support:** Compatible with a wide range of large language models, including private models (DeepSeek, Llama, Qwen) and public models (OpenAI, Claude, Gemini).
*   **Multimodal Capabilities:** Native support for processing and generating text, images, audio, and video.

## Quick Start with Docker

Get started with MaxKB in seconds using Docker:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the MaxKB web interface at `http://your_server_ip:8080` using the following default credentials:

*   Username: `admin`
*   Password: `MaxKB@123..`

For Chinese users experiencing Docker image pull failures, please refer to the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/).

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

Licensed under The GNU General Public License version 3 (GPLv3). You can find the full license at:  <https://www.gnu.org/licenses/gpl-3.0.html>