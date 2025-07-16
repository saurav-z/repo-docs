<!-- DOCTOC SKIP -->

<h1 align="center">
  <a href="https://www.skyvern.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
      <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
    </picture>
  </a>
  <br/>
  Automate Browser Workflows with the Power of LLMs and Computer Vision
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"/></a>
</p>

Skyvern is your all-in-one solution for automating complex browser-based workflows using Large Language Models (LLMs) and computer vision, replacing the need for brittle, code-heavy automation scripts.  [Explore the original repository](https://github.com/Skyvern-AI/skyvern) for the core code.

**Key Features:**

*   **Intelligent Automation:** Automate workflows on any website without needing to write custom code for each site.
*   **Resilient to Change:** Adapts to website layout changes with its vision LLM-powered approach.
*   **Multi-Website Application:** Apply a single workflow across numerous websites.
*   **Advanced Reasoning:** Leverages LLMs for complex interactions, such as interpreting driver's license age or understanding product variations.
*   **Workflow Builder:** Chaining multiple tasks together.
*   **Livestreaming:** Seeing the browser in real-time.
*   **Form Filling, Data Extraction, and File Downloading:** Automated extraction capabilities.
*   **2FA Support:** TOTP, email, and SMS-based 2FA support.
*   **Password Manager Integrations:** Integrate your password managers.

**Real-world application examples:**

*   Invoice Downloading
*   Job Application Automation
*   Materials Procurement
*   Government Website Navigation & Form Filling
*   Contact Form Automation
*   Insurance Quote Retrieval

Want to see Skyvern in action? Jump to [#real-world-examples-of-skyvern](#real-world-examples-of-skyvern)

**Quickstart**

**1. Installation:**

```bash
pip install skyvern
```

**2. Run Skyvern (choose one):**

*   **Skyvern Cloud (Managed):**

    *   Visit [app.skyvern.com](https://app.skyvern.com) to create an account.
*   **Local Installation**

    *   *UI (Recommended)*

        ```bash
        skyvern run all
        ```
        Then go to http://localhost:8080 and use the UI to run a task.

    *   *Code*
        ```python
        from skyvern import Skyvern

        skyvern = Skyvern()
        task = await skyvern.run_task(prompt="Find the top post on hackernews today")
        print(task)
        ```
        * Skyvern starts running the task in a browser that pops up and closes it when the task is done. You will be able to view the task from http://localhost:8080/history
        *   *Run on Skyvern Cloud*
            ```python
            from skyvern import Skyvern

            # Run on Skyvern Cloud
            skyvern = Skyvern(api_key="SKYVERN API KEY")

            # Local Skyvern service
            skyvern = Skyvern(base_url="http://localhost:8000", api_key="LOCAL SKYVERN API KEY")

            task = await skyvern.run_task(prompt="Find the top post on hackernews today")
            print(task)
            ```

**How it Works**

Skyvern uses a swarm of autonomous agents, inspired by BabyAGI and AutoGPT, that leverages browser automation with libraries like Playwright.

*   **Website Comprehension:** Agents understand website elements and their functions.
*   **Action Planning and Execution:** Based on your requests, Skyvern plans and executes the necessary actions.
*   **Adaptability:** Avoids hardcoded selectors and adapts to dynamic websites.

A detailed technical report can be found [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

**Demo**

[Demo](https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f)

**Performance & Evaluation**

Skyvern achieved a 64.4% accuracy on the [WebBench benchmark](webbench.ai).

**Advanced Usage**

*   **Control Your Own Browser (Chrome):** Specify the Chrome executable path.
*   **Run with Remote Browser:** Use a CDP connection URL.
*   **Consistent Output Schema:** Use `data_extraction_schema` to get structured output.
*   **Helpful Commands to Debug issues:**
    *   Launch the Skyvern Server Separately
        ```bash
        skyvern run server
        ```
    *   Launch the Skyvern UI
        ```bash
        skyvern run ui
        ```
    *   Check status of the Skyvern service
        ```bash
        skyvern status
        ```
    *   Stop the Skyvern service
        ```bash
        skyvern stop all
        ```
    *   Stop the Skyvern UI
        ```bash
        skyvern stop ui
        ```
    *   Stop the Skyvern Server Separately
        ```bash
        skyvern stop server
        ```

**Docker Compose Setup**

1.  Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.
2.  Make sure you don't have postgres running locally (Run `docker ps` to check).
3.  Clone the repository and navigate to the root directory.
4.  Run `skyvern init llm` to generate a `.env` file. This will be copied into the Docker image.
5.  Fill in the LLM provider key on the [docker-compose.yml](./docker-compose.yml). *If you want to run Skyvern on a remote server, make sure you set the correct server ip for the UI container in [docker-compose.yml](./docker-compose.yml).*
6.  Run `docker compose up -d`.
7.  Access the UI at `http://localhost:8080`.
    *Important:* If you switch from the CLI-managed Postgres to Docker Compose, you must first remove the original container:

    ```bash
    docker rm -f postgresql-container
    ```

**Skyvern Features**

*   **Tasks:** Fundamental building blocks for website interactions. Each task specifies a URL, prompt, optional data schema, and error codes.
*   **Workflows:** Chain multiple tasks together for complex automation. Supported workflow features include:
    1.  Navigation
    2.  Action
    3.  Data Extraction
    4.  Loops
    5.  File parsing
    6.  Uploading files to block storage
    7.  Sending emails
    8.  Text Prompts
    9.  Tasks (general)
    10. (Coming soon) Conditionals
    11. (Coming soon) Custom Code Block
*   **Livestreaming:** View Skyvern's browser actions in real-time.
*   **Form Filling:** Automated form completion on websites.
*   **Data Extraction:** Extract structured data from websites.
*   **File Downloading:** Automatic file downloads.
*   **Authentication:** Support for various authentication methods, including 2FA.

**Model Context Protocol (MCP)**

Skyvern supports the Model Context Protocol (MCP) to allow you to use any LLM that supports MCP.
See the MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)

**Zapier / Make.com / N8N Integration**

Skyvern supports Zapier, Make.com, and N8N to allow you to connect your Skyvern workflows to other apps.

*   [Zapier](https://docs.skyvern.com/integrations/zapier)
*   [Make.com](https://docs.skyvern.com/integrations/make.com)
*   [N8N](https://docs.skyvern.com/integrations/n8n)

**Supported LLMs**
See supported LLMs and relevant configuration details at the bottom of the original readme.

**Feature Roadmap**

See the roadmap for upcoming features.

**Contributing**

Contributions are welcome! Please refer to the [contribution guide](CONTRIBUTING.md) and  ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) to get started!

**Telemetry**

Skyvern collects basic usage statistics. Opt-out by setting the `SKYVERN_TELEMETRY` environment variable to `false`.

**License**

Licensed under the [AGPL-3.0 License](LICENSE).

**Star History**

[Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)