<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
  <br />
  Skyvern: Automate any browser-based workflow with AI.
  <br />
  <a href="https://github.com/Skyvern-AI/skyvern">
    <img src="https://img.shields.io/github/stars/skyvern-ai/skyvern?style=social" alt="GitHub Stars"/>
  </a>
  <a href="https://twitter.com/skyvernai">
    <img src="https://img.shields.io/twitter/follow/skyvernai?style=social&logo=twitter" alt="Follow on Twitter"/>
  </a>
</h1>

[Skyvern](https://github.com/Skyvern-AI/skyvern) is an open-source platform that leverages Large Language Models (LLMs) and computer vision to automate browser-based workflows, making complex web automation tasks easier than ever.  Replace brittle automation solutions with AI-powered interactions.

**Key Features:**

*   🐉 **AI-Powered Automation:** Use LLMs to understand and interact with websites, eliminating the need for custom scripts.
*   🌐 **Website Agnostic:**  Works on websites you've never seen before, adapting to layout changes with ease.
*   🔄 **Workflow Automation:**  Chain multiple tasks together to automate complex processes.
*   👁️ **Computer Vision:**  Utilizes computer vision to identify and interact with visual elements on web pages.
*   🚀 **SOTA Performance:** Achieves state-of-the-art performance in browser automation tasks (see our [technical report](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/)).
*   🔒 **Secure & Customizable:**  Supports 2FA, password manager integrations, and offers a flexible, modular design.
*   💻 **Open Source:**  Access all the core logic with the  [AGPL-3.0 License](LICENSE).

**How it Works:**

Skyvern employs a swarm of AI agents, inspired by the task-driven approach of  [BabyAGI](https://github.com/yoheinakajima/babyagi) and  [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT). This allows Skyvern to:

1.  Comprehend a website through computer vision and planning
2.  Execute actions using browser automation libraries like [Playwright](https://playwright.dev/).

Key Advantages:

*   **Adaptability:** Operates on unseen websites without custom code.
*   **Resilience:**  Unaffected by website layout changes.
*   **Scalability:**  Applies workflows across numerous websites.
*   **Intelligence:**  LLMs reason through complex scenarios (e.g., insurance quotes, product comparisons).

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Geico Automation Demo"/>
</p>

**Demo:**

See Skyvern in action! [Check out the demos](https://app.skyvern.com/tasks/create).

**Quickstart:**

1.  **Install:** `pip install skyvern`
2.  **Run Quickstart (for setup):** `skyvern quickstart`
3.  **Run Server & UI (Recommended):** `skyvern run all`
4.  **Run task:**
    ```python
    from skyvern import Skyvern

    skyvern = Skyvern()
    task = await skyvern.run_task(prompt="Find the top post on hackernews today")
    print(task)
    ```
    (Access the UI at  `http://localhost:8080` to view your tasks)

**Detailed setup instructions and advanced usage** can be found in the full README.

**Documentation:**

Comprehensive documentation is available on our  [docs page](https://docs.skyvern.com).

**Join the Community:**

*   [Discord](https://discord.gg/fG2XXEuQX3)
*   [Email](mailto:founders@skyvern.com)

**Contributing:**

We welcome contributions! See our  [contribution guide](CONTRIBUTING.md) and  ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

**Star History:**

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)
```

**Key improvements and SEO optimization:**

*   **Strong Headline:** The hook now stands out, attracting user attention.
*   **Keywords:** Incorporated relevant keywords like "browser automation," "LLMs," "computer vision," "web automation," and "AI-powered" throughout the summary.
*   **Clear Structure:**  Used headings, bullet points, and concise language for readability.
*   **Call to Action:**  Provides clear calls to action with the demo and documentation links.
*   **SEO-Friendly:**  Optimized the text for search engines with key phrases related to browser automation and AI.
*   **Concise and Focused:** Removed unnecessary details and focused on the most important features and benefits.
*   **Visual Appeal:** Kept the images for visual interest and understanding.
*   **Included relevant links to various sources.**
*   **Star History** Included a star history chart to improve SEO.