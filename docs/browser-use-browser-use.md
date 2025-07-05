<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Give Your AI the Power of a Web Browser 🤖</h1>

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

**Browser Use empowers your AI agents to interact with the web, automating tasks and unlocking new possibilities.**  This allows you to connect your AI agents with the browser in the easiest way possible!

*   **Key Features:**
    *   Effortless browser automation via AI control.
    *   Simple setup and integration with existing AI models.
    *   Ready-to-use examples for common tasks.
    *   Hosted cloud version for instant use.
    *   Interactive CLI for easy testing and development.
    *   Robust testing framework to validate your agents.

*   Join the community in our [Discord](https://link.browser-use.com/discord) and show off your project!  Check out our [Merch store](https://browsermerch.com) for cool swag!

*   **Cloud Access:** Skip the setup and try our **hosted version** for instant browser automation! **[Try the cloud ☁︎](https://cloud.browser-use.com)**.

## Getting Started

1.  **Installation (Python):**
    ```bash
    pip install browser-use
    ```

2.  **Install Browser:**
    ```bash
    playwright install chromium --with-deps --no-shell
    ```

3.  **Example Usage:**
    ```python
    import asyncio
    from dotenv import load_dotenv
    load_dotenv()
    from browser_use import Agent
    from browser_use.llm import ChatOpenAI

    async def main():
        agent = Agent(
            task="Compare the price of gpt-4o and DeepSeek-V3",
            llm=ChatOpenAI(model="gpt-4o"),
        )
        await agent.run()

    asyncio.run(main())
    ```

4.  **API Keys:** Add your API keys to your `.env` file.
    ```bash
    OPENAI_API_KEY=
    ANTHROPIC_API_KEY=
    AZURE_OPENAI_ENDPOINT=
    AZURE_OPENAI_KEY=
    GOOGLE_API_KEY=
    DEEPSEEK_API_KEY=
    GROK_API_KEY=
    NOVITA_API_KEY=
    ```

    For other settings, models, and more, check out the comprehensive [documentation 📕](https://docs.browser-use.com).

## Test & Explore

*   **Web UI & Desktop App:** Test browser-use with our [Web UI](https://github.com/browser-use/web-ui) or [Desktop App](https://github.com/browser-use/desktop).

*   **Interactive CLI:**  Try the `browser-use` interactive CLI:

    ```bash
    pip install "browser-use[cli]"
    browser-use
    ```

## Use Cases & Demos

Explore practical applications with these demos:

<br/>

*   **Shopping:** Add grocery items to cart and checkout.
    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

<br/>

*   **LinkedIn to Salesforce:**
    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

<br/>

*   **Job Application:** Read my CV & find ML jobs, save them to a file, and then start applying for them in new tabs, if you need help, ask me.
    https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

<br/>

*   **Google Docs:** Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF.
    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

<br/>

*   **Hugging Face:** Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.
    https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

<br/>

*   For more examples, see the [examples](examples) folder or the [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repository for inspiration.

## Project Vision

Make your computer do what you tell it to do using the power of AI and web browsers!

## Development Roadmap

### Agent

*   [ ] Improve agent memory to handle +100 steps
*   [ ] Enhance planning capabilities (load website specific context)
*   [ ] Reduce token consumption (system prompt, DOM state)

### DOM Extraction

*   [ ] Enable detection for all possible UI elements
*   [ ] Improve state representation for UI elements so that all LLMs can understand what's on the page

### Workflows

*   [ ] Let user record a workflow - which we can rerun with browser-use as a fallback
*   [ ] Make rerunning of workflows work, even if pages change

### User Experience

*   [ ] Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.
*   [ ] Improve docs
*   [ ] Make it faster

### Parallelization

*   [ ] Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.

## Contribute

Contributions are welcome!  Please open issues for bugs or feature requests.  For documentation contributions, check out the `/docs` folder.

## Robust Agent Testing 🧪

Ensure your agents are resilient by running your tasks automatically in our CI on every update!

*   **Add Your Task:** Create a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automated Validation:** Every update triggers your task and evaluates it using your defined criteria.

## Local Setup

Learn more about the library with the [local setup 📕](https://docs.browser-use.com/development/local-setup).

**Important Note:** `main` is the primary development branch. For production use, install a [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag & More!

Show off your Browser-use love with our [Merch store](https://browsermerch.com)!  Good contributors get free swag! 👀

## Citation

If you use Browser Use in your research or project, please cite:

```bibtex
@software{browser_use2024,
  author = {Müller, Magnus and Žunič, Gregor},
  title = {Browser Use: Enable AI to control your browser},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/browser-use/browser-use}
}
```

 <div align="center"> <img src="https://github.com/user-attachments/assets/06fa3078-8461-4560-b434-445510c1766f" width="400"/>

[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)

 </div>

<div align="center">
Made with ❤️ in Zurich and San Francisco
 </div>
```

Key improvements and SEO considerations:

*   **Clear, concise, and engaging title:**  "Browser Use: Give Your AI the Power of a Web Browser 🤖"  immediately conveys the project's purpose.
*   **Strong introductory sentence:**  The first sentence immediately hooks the reader.
*   **Keyword Optimization:**  The README uses terms like "AI," "browser automation," "web browser,"  and "AI agents" to target relevant search queries.
*   **Clear Structure with Headings:** Well-organized headings make the content easy to scan.
*   **Bulleted Key Features:** Highlight essential features for quick understanding.
*   **Concise & Benefit-Oriented Descriptions:**  Features are explained succinctly and focus on the value to the user.
*   **Call to Action (CTA):**  Encourages community participation and merch store visits.
*   **Comprehensive "Getting Started" Section:** Includes installation steps and a basic code example.
*   **Emphasis on Demos and Use Cases:** Showcases practical applications to increase engagement.
*   **Clear Roadmap & Contributing Guidelines:**  Encourages contribution and transparency.
*   **Robust Testing Section:** Unique selling point, highlights automated validation.
*   **Citation Information:**  Provides guidance for proper attribution.
*   **Visual Appeal:** The original image, social media badges, and "Made with ❤️" section add visual interest.
*   **Links to original repo are retained.**
*   **Cloud Usage Emphasized:** The cloud offering is promoted at the start and in the Quick Start section.