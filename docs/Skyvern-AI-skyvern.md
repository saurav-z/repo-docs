<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png" alt="Skyvern Logo"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
  <br />
  <br />
  Automate any browser workflow with the power of AI.
  <br />
  <br />
  <a href="https://github.com/Skyvern-AI/skyvern">
    <img src="https://img.shields.io/github/stars/skyvern-ai/skyvern?style=social" alt="GitHub stars"/>
  </a>
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Docs"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"/></a>
</p>

Skyvern is a cutting-edge AI-powered automation tool that allows you to automate complex browser-based workflows using Large Language Models (LLMs) and computer vision.  This powerful combination eliminates the need for brittle, hard-coded scripts by enabling Skyvern to understand and interact with websites, even those it's never seen before.

**Key Features:**

*   🚀 **Automated Browser Automation:**  Control and interact with any website.
*   🤖 **AI-Driven Interactions:** Leverages LLMs to understand and execute tasks.
*   🌐 **Website Agnostic:**  Works on websites without requiring custom code.
*   🔄 **Resilient to Changes:**  Adapts to website layout changes automatically.
*   🔗 **Workflow Creation:**  Chain multiple tasks together for complex automation.
*   👁️ **Livestreaming:** See exactly what Skyvern is doing.
*   📝 **Form Filling & Data Extraction:** Automate form submissions and extract data.
*   📥 **File Handling:** Download files from websites and upload them to cloud storage.
*   🔐 **Authentication:**  Supports various authentication methods, including 2FA.
*   🔌 **Integrations:** Zapier, Make.com, N8N, Model Context Protocol.

**Getting Started**

1.  **Install:**
    ```bash
    pip install skyvern
    ```

2.  **Run a Quick Task (UI):**
    ```bash
    skyvern run all
    ```
    Then go to http://localhost:8080 and use the UI to run a task

3.  **Run a Quick Task (Code):**
    ```python
    from skyvern import Skyvern

    skyvern = Skyvern()
    task = await skyvern.run_task(prompt="Find the top post on hackernews today")
    print(task)
    ```
    The above command runs a task in a browser that pops up and closes it when the task is done. You will be able to view the task from http://localhost:8080/history

**Explore [Skyvern on GitHub](https://github.com/Skyvern-AI/skyvern) for detailed installation instructions, code examples, advanced usage, and contributing guidelines.**