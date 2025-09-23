<p align="center"><img src= "https://github.com/1Panel-dev/maxkb/assets/52996290/c0694996-0eed-40d8-b369-322bf2a380bf" alt="MaxKB" width="300" /></p>

## MaxKB: Build Powerful Enterprise-Grade AI Agents with Ease

MaxKB is an open-source platform that empowers you to build cutting-edge, enterprise-ready AI agents with advanced features and seamless integration.

<p align="center"><a href="https://trendshift.io/repositories/9113" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9113" alt="1Panel-dev%2FMaxKB | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a></p>
<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html#license-text"><img src="https://img.shields.io/github/license/1Panel-dev/maxkb?color=%231890FF" alt="License: GPL v3"></a>
  <a href="https://github.com/1Panel-dev/maxkb/releases/latest"><img src="https://img.shields.io/github/v/release/1Panel-dev/maxkb" alt="Latest release"></a>
  <a href="https://github.com/1Panel-dev/maxkb"><img src="https://img.shields.io/github/stars/1Panel-dev/maxkb?color=%231890FF&style=flat-square" alt="Stars"></a>    
  <a href="https://hub.docker.com/r/1panel/maxkb"><img src="https://img.shields.io/docker/pulls/1panel/maxkb?label=downloads" alt="Download"></a><br/>
 [<a href="/README_CN.md">中文(简体)</a>] | [<a href="/README.md">English</a>] 
</p>
<hr/>

**[Explore the MaxKB Repository on GitHub](https://github.com/1Panel-dev/MaxKB)**

MaxKB (Max Knowledge Brain) provides a robust and user-friendly platform for creating AI agents tailored for various enterprise applications. It integrates Retrieval-Augmented Generation (RAG) pipelines, supports complex workflows, and offers powerful tool-use capabilities. Perfect for intelligent customer service, internal knowledge bases, research, and education.

### Key Features

*   **RAG Pipeline:**
    *   Supports direct document uploads and web crawling.
    *   Includes automatic text splitting and vectorization for efficient knowledge retrieval.
    *   Reduces hallucinations in large language models (LLMs) for improved accuracy.
*   **Agentic Workflow:**
    *   Offers a powerful workflow engine for orchestrating AI processes.
    *   Provides a comprehensive function library and MCP tool-use.
    *   Enables automation to meet the demands of complex business needs.
*   **Seamless Integration:**
    *   Facilitates zero-coding integration with third-party systems.
    *   Quickly equips existing systems with smart Q&A features.
    *   Enhances user satisfaction by improving access to information.
*   **Model-Agnostic:**
    *   Compatible with a wide range of LLMs, including private models (e.g., DeepSeek, Llama, Qwen) and public models (e.g., OpenAI, Claude, Gemini).
*   **Multi-Modal Support:**
    *   Native support for text, image, audio, and video input and output.

## Quick Start with Docker

Get started with MaxKB using the following Docker command:

```bash
docker run -d --name=maxkb --restart=always -p 8080:8080 -v ~/.maxkb:/opt/maxkb 1panel/maxkb
```

Access the MaxKB web interface at `http://your_server_ip:8080` using the default credentials:

*   **Username:** admin
*   **Password:** MaxKB@123..

**Note for Chinese Users:** If you encounter issues pulling the Docker image, please refer to the [offline installation documentation](https://maxkb.cn/docs/v2/installation/offline_installtion/) for alternative installation instructions.

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

Licensed under The GNU General Public License version 3 (GPLv3).  You can find the full license text at:

<https://www.gnu.org/licenses/gpl-3.0.html>

The software is provided "AS IS" without warranties or conditions.
```
Key improvements and SEO considerations:

*   **Clear, concise title:** "MaxKB: Build Powerful Enterprise-Grade AI Agents with Ease" clearly states the project's purpose and target audience.
*   **One-sentence hook:**  The first sentence immediately grabs the reader's attention and summarizes the core value proposition.
*   **Keyword optimization:** Keywords like "AI agents," "enterprise-grade," "RAG," "LLMs" are included naturally throughout the description.
*   **Bulleted key features:**  Features are presented in a clear, scannable format for quick understanding.
*   **Actionable Quick Start:** The Docker command is highlighted, making it easy for users to get started.
*   **Descriptive Headings:**  Headings are used to improve readability and organization.
*   **Internal and external links:**  Links to the GitHub repo, technologies, and the license are included for better SEO.
*   **Concise language:** The text is clear, concise, and avoids unnecessary jargon.
*   **Alt text for images:**  All images have descriptive alt text for accessibility and SEO.
*   **Clear call to action:** The "Explore the MaxKB Repository on GitHub" link encourages users to learn more.