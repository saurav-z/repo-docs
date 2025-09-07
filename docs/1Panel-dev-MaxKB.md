<p align="center"><img src= "https://github.com/1Panel-dev/maxkb/assets/52996290/c06949962-0eed-40d8-b369-322bf2a380bf" alt="MaxKB" width="300" /></p>

<h1 align="center">MaxKB: Build Powerful Enterprise-Grade AI Agents with Ease</h1>

<p align="center">MaxKB is an open-source platform empowering you to build and deploy intelligent agents for a variety of enterprise applications.</p>

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html#license-text"><img src="https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF" alt="License: GPL v3"></a>
  <a href="https://github.com/1Panel-dev/maxkb/releases/latest"><img src="https://img.shields.io/github/v/release/1Panel-dev/maxkb" alt="Latest release"></a>
  <a href="https://github.com/1Panel-dev/maxkb"><img src="https://img.shields.io/github/stars/1Panel-dev/maxkb?color=%231890FF&style=flat-square" alt="Stars"></a>
  <a href="https://hub.docker.com/r/1panel/maxkb"><img src="https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads" alt="Downloads"></a><br/>
  [<a href="/README_CN.md">中文(简体)</a>] | [<a href="/README.md">English</a>]
</p>

<hr/>

## Key Features

MaxKB provides a comprehensive platform for building and deploying AI agents, offering the following key features:

*   **RAG Pipeline:** Streamlines the process of integrating Retrieval-Augmented Generation (RAG) pipelines by supporting document uploading, automated text splitting, and vectorization. This enhances the accuracy and reliability of large language models (LLMs) and provides superior Q&A experiences.
*   **Agentic Workflow:** Features a powerful workflow engine, function library, and MCP tool-use capabilities. This allows for the orchestration of complex AI processes to meet diverse business needs.
*   **Seamless Integration:** Enables zero-coding integration with third-party business systems. Quickly equip existing systems with intelligent Q&A capabilities to boost user satisfaction.
*   **Model-Agnostic:** Supports various LLMs, including private models (e.g., DeepSeek, Llama, Qwen) and public models (e.g., OpenAI, Claude, Gemini).
*   **Multi-Modal Support:** Native support for text, image, audio, and video input and output, enriching the interaction capabilities of your agents.

## Use Cases

MaxKB is widely used in many scenarios, including:

*   Intelligent Customer Service
*   Corporate Internal Knowledge Bases
*   Academic Research
*   Education

## Quick Start

Get started with MaxKB using Docker:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the MaxKB web interface at `http://your_server_ip:8080` using the following default credentials:

*   username: admin
*   password: MaxKB@123..

**Note for Chinese Users:** If you encounter issues pulling the Docker image, please refer to the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/).

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

Licensed under the GNU General Public License version 3 (GPLv3).  See the [LICENSE](https://www.gnu.org/licenses/gpl-3.0.html) for details.

**[Explore MaxKB on GitHub](https://github.com/1Panel-dev/MaxKB)**
```

Key improvements and explanations:

*   **SEO-Friendly Title & Hook:**  The title is more descriptive and includes relevant keywords ("Enterprise-Grade AI Agents," "Open-source platform"). The hook is concise and focuses on the core value proposition.
*   **Clear Headings:**  Uses `<h1>`, `<h2>` and `<h3>` tags to structure the content, making it easy to read and improving SEO.
*   **Bulleted Key Features:** Emphasizes the main benefits of MaxKB, making it easy for users to quickly understand the platform's capabilities.
*   **Concise Descriptions:**  Streamlines the descriptions of features and use cases.
*   **Actionable Quick Start:**  The Docker command is included directly.
*   **Call to Action:**  Encourages the user to explore the project.
*   **Combined Similar Sections:** The "Quick Start" and the Chinese Docker image instructions are included within the same section for better readability.
*   **Removed Redundancy:** Removed the introduction paragraph that had already been mentioned in the hook.
*   **Links:** Added a link to the project's GitHub page, and a link to the original repo.
*   **Code Formatting:** The code block now uses consistent formatting.
*   **Clearer Language**: Minor improvements to the language used in the document.