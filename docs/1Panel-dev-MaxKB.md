<p align="center">
  <img src="https://github.com/1Panel-dev/maxkb/assets/52996290/c0694996-0eed-40d8-b369-322bf2a380bf" alt="MaxKB" width="300" />
</p>

<h1 align="center">MaxKB: Build Enterprise-Grade AI Agents with Ease</h1>

MaxKB is an open-source platform empowering you to build powerful AI agents for a variety of enterprise applications. (See the original repository on GitHub: [1Panel-dev/MaxKB](https://github.com/1Panel-dev/MaxKB)).

<p align="center">
  <a href="https://trendshift.io/repositories/9113" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/9113" alt="1Panel-dev%2FMaxKB | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</p>

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html#license-text">
    <img src="https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF" alt="License: GPL v3">
  </a>
  <a href="https://github.com/1Panel-dev/maxkb/releases/latest">
    <img src="https://img.shields.io/github/v/release/1Panel-dev/maxkb" alt="Latest release">
  </a>
  <a href="https://github.com/1Panel-dev/maxkb">
    <img src="https://img.shields.io/github/stars/1Panel-dev/maxkb?color=%231890FF&style=flat-square" alt="Stars">
  </a>
  <a href="https://hub.docker.com/r/1panel/maxkb">
    <img src="https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads" alt="Download">
  </a><br/>
  [<a href="/README_CN.md">中文(简体)</a>] | [<a href="/README.md">English</a>]
</p>

<hr/>

## Key Features

MaxKB, or Max Knowledge Brain, provides a robust and versatile platform for building advanced AI agents tailored for enterprise needs. Key features include:

*   **RAG Pipeline:** Seamlessly integrates Retrieval-Augmented Generation (RAG) pipelines for improved accuracy and reduced hallucinations. Includes document uploading, automatic text splitting, and vectorization.
*   **Agentic Workflow:** Offers a powerful workflow engine with function libraries and MCP tool-use, enabling complex AI process orchestration.
*   **Easy Integration:** Rapidly integrates with existing systems, providing intelligent Q&A capabilities without extensive coding.
*   **Model Agnostic:** Supports a wide range of Large Language Models (LLMs), including private models (DeepSeek, Llama, Qwen, etc.) and public models (OpenAI, Claude, Gemini, etc.).
*   **Multi Modal Support:** Native support for processing and generating text, images, audio, and video.

## Quick Start with Docker

Get started with MaxKB quickly using Docker:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the web interface at `http://your_server_ip:8080` using the default credentials:

*   username: admin
*   password: MaxKB@123..

**Note for Chinese Users:** If you encounter Docker image pull issues, please refer to the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/) for installation instructions.

## Screenshots

<table style="border-collapse: collapse; border: 1px solid black;">
  <tr>
    <td style="padding: 5px;background-color:#fff;">
      <img src="https://github.com/user-attachments/assets/eb285512-a66a-4752-8941-c65ed1592238" alt="MaxKB Demo1" />
    </td>
    <td style="padding: 5px;background-color:#fff;">
      <img src="https://github.com/user-attachments/assets/f732f1f5-472c-4fd2-93c1-a277eda83d04" alt="MaxKB Demo2" />
    </td>
  </tr>
  <tr>
    <td style="padding: 5px;background-color:#fff;">
      <img src="https://github.com/user-attachments/assets/c927474a-9a23-4830-822f-5db26025c9b2" alt="MaxKB Demo3" />
    </td>
    <td style="padding: 5px;background-color:#fff;">
      <img src="https://github.com/user-attachments/assets/e6268996-a46d-4e58-9f30-31139df78ad2" alt="MaxKB Demo4" />
    </td>
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

Licensed under The GNU General Public License version 3 (GPLv3). See the license details at:

<https://www.gnu.org/licenses/gpl-3.0.html>

This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
```
Key improvements and SEO considerations:

*   **Headline Optimization:**  Added a strong, keyword-rich H1 heading that includes the core value proposition (Build Enterprise-Grade AI Agents).
*   **Concise Hook:** The opening sentence provides a clear and compelling value proposition.
*   **Keyword Integration:** Integrated relevant keywords like "AI agents," "enterprise," "RAG," "LLMs," "workflow," etc., naturally throughout the description.
*   **Structured Content:** Uses clear headings (Key Features, Quick Start, etc.) for improved readability and SEO.
*   **Bulleted Lists:** Emphasizes key features, making them easy to scan and understand.
*   **Internal and External Linking:** Keeps original links and adds link to github in the first paragraph.
*   **Context and Value:** Provides context around the features, explaining their benefits.
*   **Call to Action:** The "Quick Start" section encourages immediate engagement.
*   **Alt Text:** Ensured all images have descriptive alt text.
*   **Clear Language:**  Uses straightforward and easy-to-understand language.
*   **Responsive Design**: No changes needed since this is markdown but important to consider for website implementations.