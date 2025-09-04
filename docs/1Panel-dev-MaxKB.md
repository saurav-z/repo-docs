<p align="center">
  <img src="https://github.com/1Panel-dev/maxkb/assets/52996290/c0694996-0eed-40d8-b369-322bf2a380bf" alt="MaxKB" width="300" />
</p>

<h1 align="center">MaxKB: Build Powerful Enterprise AI Agents with Ease</h1>

**MaxKB** empowers you to create intelligent, enterprise-grade AI agents with its open-source platform. This comprehensive platform integrates Retrieval-Augmented Generation (RAG) pipelines, robust workflows, and advanced tool use capabilities, designed for a variety of applications, including intelligent customer service, knowledge management, and research.  Discover more on the [original repository](https://github.com/1Panel-dev/MaxKB).

<p align="center">
  <a href="https://trendshift.io/repositories/9113" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9113" alt="1Panel-dev%2FMaxKB | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>
<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html#license-text"><img src="https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF" alt="License: GPL v3"></a>
  <a href="https://github.com/1Panel-dev/maxkb/releases/latest"><img src="https://img.shields.io/github/v/release/1Panel-dev/maxkb" alt="Latest release"></a>
  <a href="https://github.com/1Panel-dev/maxkb"><img src="https://img.shields.io/github/stars/1Panel-dev/maxkb?color=%231890FF&style=flat-square" alt="Stars"></a>
  <a href="https://hub.docker.com/r/1panel/maxkb"><img src="https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads" alt="Download"></a><br/>
  [<a href="/README_CN.md">中文(简体)</a>] | [<a href="/README.md">English</a>]
</p>

## Key Features

*   **RAG Pipeline:**  Effortlessly upload documents or crawl online content. Features automatic text splitting and vectorization to enhance large language model (LLM) accuracy, leading to superior Q&A experiences and reduced hallucinations.
*   **Agentic Workflow:**  Utilize a powerful workflow engine with a function library and MCP tool-use, enabling the orchestration of AI processes to handle complex business scenarios.
*   **Seamless Integration:**  Integrate into third-party systems with minimal coding, rapidly equipping existing systems with smart Q&A capabilities to boost user satisfaction.
*   **Model-Agnostic:**  Supports a wide array of large language models, including private models (DeepSeek, Llama, Qwen) and public models (OpenAI, Claude, Gemini).
*   **Multi-Modal Support:**  Native support for diverse data types, including text, images, audio, and video, enhancing agent versatility.

## Getting Started

Get up and running with MaxKB quickly using Docker:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the MaxKB web interface at `http://your_server_ip:8080` with the following default credentials:

*   username: admin
*   password: MaxKB@123..

**Note for Chinese Users:**  If you experience Docker image pull failures, please refer to the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/) for assistance.

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

## Technology Stack

*   **Frontend:** [Vue.js](https://vuejs.org/)
*   **Backend:** [Python / Django](https://www.djangoproject.com/)
*   **LLM Framework:** [LangChain](https://www.langchain.com/)
*   **Database:** [PostgreSQL + pgvector](https://www.postgresql.org/)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=1Panel-dev/MaxKB&type=Date)](https://star-history.com/#1Panel-dev/MaxKB&Date)

## License

Licensed under The GNU General Public License version 3 (GPLv3). You can find the license details at:  <https://www.gnu.org/licenses/gpl-3.0.html>

This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
```
Key improvements and explanations:

*   **SEO Optimization:**  The title uses keywords like "Enterprise AI Agents," "Open-source," and "RAG."  The headings are clear and use relevant keywords.
*   **One-Sentence Hook:** The second sentence provides a strong and concise overview of what MaxKB is.
*   **Clear Structure:** The document is well-structured with clear headings, making it easy to read and navigate.
*   **Bulleted Key Features:** This format makes it easy for users to quickly grasp the core functionalities.
*   **Concise Language:**  The text is more concise and avoids unnecessary jargon.
*   **Call to Action:**  Includes a clear call to action to try out the project.
*   **Emphasis on Benefits:** Highlights the advantages of using MaxKB (e.g., reducing hallucinations, enhancing user satisfaction).
*   **Improved Formatting:**  Consistent use of bolding and bullet points.
*   **Complete and Self-Contained:** The rewritten README provides all the essential information.
*   **Includes all original information:** All original content has been incorporated into the new README.
*   **Clearer Introduction:** The introduction now is easier to understand.
*   **Revised Sections:** Sections are more descriptive.
*   **Removed Redundancy:** Removed redundant phrases.
*   **Markdown Formatting:**  The entire README is properly formatted in Markdown for better readability and rendering on GitHub.