<h1 align="center">
  Skyvern: Automate Browser Workflows with LLMs and Computer Vision
</h1>

<p align="center">
  <a href="https://www.skyvern.com/">
    <img src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo" height="120">
  </a>
</p>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" /></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin"/></a>
</p>

**Tired of repetitive browser tasks?**  Skyvern uses the power of Large Language Models (LLMs) and computer vision to automate complex browser-based workflows, making web automation easier and more reliable than ever before. Explore the [Skyvern](https://github.com/Skyvern-AI/skyvern) repository to get started.

**Key Features:**

*   🐉 **LLM-Powered Automation:** Automates browser interactions using LLMs, eliminating the need for brittle, hard-coded scripts.
*   👁️ **Computer Vision:**  Understands and interacts with websites based on visual elements, adaptable to layout changes.
*   🌐 **Cross-Website Compatibility:**  Applies workflows across numerous websites without custom code for each.
*   🦾 **Robust Workflows:** Supports complex scenarios, including form filling, data extraction, file downloads, and more.
*   ☁️ **Skyvern Cloud:**  A managed cloud version for easy deployment, including anti-bot measures, proxy network, and CAPTCHA solvers.
*   💻 **Flexible Integration:** Supports various LLM providers,  Zapier/Make.com/N8N, and custom integrations.
*   🛡️ **Authentication Support:** Includes support for 2FA (TOTP), password managers, and more.
*   🗺️ **Advanced Features:**  Including Model Context Protocol (MCP), Docker Compose setup, and more.

**Get Started**

1.  **Install Skyvern:**

    ```bash
    pip install skyvern
    ```

2.  **Run Skyvern:**

    ```bash
    skyvern quickstart
    ```

3.  **Run a Task (UI):**

    ```bash
    skyvern run all
    ```

    Access the UI at [http://localhost:8080](http://localhost:8080) to run a task.

4.  **Run a Task (Code):**
   ```python
    from skyvern import Skyvern

    skyvern = Skyvern()
    task = await skyvern.run_task(prompt="Find the top post on hackernews today")
    print(task)
   ```

    Read about running tasks on different targets [here](#quickstart).

**How Skyvern Works**

Skyvern leverages a swarm of agents inspired by BabyAGI and AutoGPT, empowered by browser automation libraries like Playwright.  This approach offers key advantages:

*   Operates on unseen websites by mapping visual elements to actions.
*   Resistant to website layout changes, unlike traditional methods relying on specific selectors.
*   Applies a single workflow to numerous websites, reasoning through the necessary interactions.
*   Utilizes LLMs to handle complex situations, such as inferring answers to form questions.

See a detailed technical report [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

**Demo**

<!-- Replace with a more compelling demo image or video. -->
<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif"/>
</p>

**Performance & Evaluation**

Skyvern achieves state-of-the-art (SOTA) performance on the [WebBench benchmark](webbench.ai), with 64.4% accuracy.

*   **WRITE Tasks (RPA):** Skyvern excels at "WRITE" tasks (form filling, downloads, etc.).

    <p align="center">
        <img src="fern/images/performance/webbench_write.png"/>
    </p>

**[View the original repository for more details](https://github.com/Skyvern-AI/skyvern)**
```
Key Improvements and Summary of Changes:

*   **SEO Optimization:**  Included relevant keywords in headings and descriptions (e.g., "browser automation," "LLMs," "computer vision," "web workflows," "RPA").
*   **Concise Hook:** A one-sentence hook to immediately grab the reader's attention.
*   **Clear Headings:**  Organized the content with clear, SEO-friendly headings (e.g., "Key Features," "Get Started," "How Skyvern Works").
*   **Bulleted Key Features:** Easier for readers to scan and understand Skyvern's core capabilities.
*   **Actionable Call to Action:** Encouraged users to explore the repository.
*   **Concise Content:** Removed redundant information and streamlined the text.
*   **Enhanced Demo Area:**  Added a placeholder for a more compelling visual demo.
*   **Better Structure and Formatting:**  Made the README more visually appealing and easier to read.
*   **Links & References:** Retained important links to the documentation, website, and other resources.
*   **Added a call to action to view the original repo.**
*   **Removed some less important details, and shortened some of the original content, making it a better overview of the product.**
*   **Included the key features that make skyvern a great product.**
*   **Included all key functionality without going into excessive detail.**

This revised README is much more effective at attracting attention, conveying the value proposition of Skyvern, and guiding users to the next steps.