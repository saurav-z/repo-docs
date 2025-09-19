<p align="center"><img src= "https://github.com/1Panel-dev/maxkb/assets/52996290/c0694996-0eed-40d8-b369-322bf2a380bf" alt="MaxKB" width="300" /></p>

<h1 align="center">MaxKB: Build Powerful Enterprise-Grade AI Agents with Ease</h1>

<p align="center">MaxKB is an open-source platform designed to empower you to build and deploy sophisticated AI agents for a variety of enterprise applications.</p>

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html#license-text"><img src="https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF" alt="License: GPL v3"></a>
  <a href="https://github.com/1Panel-dev/maxkb/releases/latest"><img src="https://img.shields.io/github/v/release/1Panel-dev/maxkb" alt="Latest release"></a>
  <a href="https://github.com/1Panel-dev/maxkb"><img src="https://img.shields.io/github/stars/1Panel-dev/maxkb?color=%231890FF&style=flat-square" alt="Stars"></a>    
  <a href="https://hub.docker.com/r/1panel/maxkb"><img src="https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads" alt="Download"></a><br/>
  [<a href="/README_CN.md">中文(简体)</a>] | [<a href="/README.md">English</a>] 
  <br/>
  <a href="https://github.com/1Panel-dev/MaxKB"> ➡️ View the original repository on GitHub</a>
</p>

<hr/>

## Key Features of MaxKB

MaxKB (Max Knowledge Brain) provides a robust and flexible platform for building intelligent agents.  Here's what makes it stand out:

*   **RAG Pipeline for Enhanced Accuracy:**  Seamlessly integrates Retrieval-Augmented Generation (RAG) pipelines.  Supports document uploading and online document crawling, complete with automatic text splitting and vectorization to reduce hallucinations.
*   **Agentic Workflow Engine:** Includes a powerful workflow engine, function library, and MCP tool-use capabilities, enabling the orchestration of AI processes for complex business scenarios.
*   **Effortless Integration:**  Facilitates rapid integration with existing systems through zero-coding, allowing you to quickly add intelligent Q&A to enhance user satisfaction.
*   **Model Agnostic:**  Compatible with a wide range of large language models (LLMs), including both private models (e.g., DeepSeek, Llama, Qwen) and public models (e.g., OpenAI, Claude, Gemini).
*   **Multimodal Support:**  Native support for diverse input and output formats, including text, images, audio, and video.

## Quick Start with Docker

Get up and running quickly with MaxKB using Docker:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the MaxKB web interface at `http://your_server_ip:8080` using the default credentials:

*   **Username:** admin
*   **Password:** MaxKB@123..

**Note for Chinese Users:**  If you encounter issues pulling the Docker image, please consult the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/) for alternative installation methods.

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

Licensed under The GNU General Public License version 3 (GPLv3).  You can find the full license details at:

<https://www.gnu.org/licenses/gpl-3.0.html>

---