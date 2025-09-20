<p align="center">
  <img src="https://resource-fit2cloud-com.oss-cn-hangzhou.aliyuncs.com/sqlbot/sqlbot.png" alt="SQLBot" width="300" />
</p>

# SQLBot: Your AI-Powered Assistant for Data Insights

**SQLBot revolutionizes data analysis by enabling users to ask questions in natural language and receive SQL queries and insights effortlessly.** This open-source tool leverages large language models (LLMs) and Retrieval-Augmented Generation (RAG) to provide a user-friendly and efficient way to interact with your data.

[Check out the original repository on GitHub!](https://github.com/dataease/SQLBot)

## Key Features:

*   **Text-to-SQL Conversion:** Transforms natural language questions into accurate SQL queries using advanced LLMs and RAG.
*   **Ease of Use:** Requires minimal setup; simply configure your data source and LLM to start querying.
*   **Seamless Integration:** Integrates with various AI platforms, including n8n, MaxKB, Dify, and Coze, to expand your application's capabilities.
*   **Secure and Controlled:** Offers workspace-based resource isolation and fine-grained data access control for enhanced security.
*   **Dockerized Deployment:** Easy deployment with a single Docker command.

## How it Works

[Image of SQLBot architecture - see original README for image]

## Getting Started

### Installation

SQLBot is easy to deploy using Docker:

1.  **Prerequisites:** Ensure you have a Linux server with Docker installed.
2.  **Run the Installation Script:**

    ```bash
    docker run -d \
      --name sqlbot \
      --restart unless-stopped \
      -p 8000:8000 \
      -p 8001:8001 \
      -v ./data/sqlbot/excel:/opt/sqlbot/data/excel \
      -v ./data/sqlbot/images:/opt/sqlbot/images \
      -v ./data/sqlbot/logs:/opt/sqlbot/app/logs \
      -v ./data/postgresql:/var/lib/postgresql/data \
      --privileged=true \
      dataease/sqlbot
    ```

    You can also deploy SQLBot using [1Panel](https://apps.fit2cloud.com/1panel) or via [offline installation packages](https://community.fit2cloud.com/#/products/sqlbot/downloads) for air-gapped environments.

### Accessing SQLBot

*   **Web Interface:** Access the SQLBot interface through your web browser at `http://<your server IP>:8000/`.
*   **Login Credentials:**
    *   Username: `admin`
    *   Password: `SQLBot@123456`

### Contact Us

Have questions or need support? Join our technical交流群.

[Image of contact QR code - see original README for image]

## UI Example

[Image of SQLBot user interface - see original README for image]

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dataease/sqlbot&type=Date)](https://www.star-history.com/#dataease/sqlbot&Date)

## Related Projects from FIT2CLOUD

*   [DataEase](https://github.com/dataease/dataease/) - Open-source BI tool
*   [1Panel](https://github.com/1panel-dev/1panel/) - Linux server management panel
*   [MaxKB](https://github.com/1panel-dev/MaxKB/) - Enterprise-grade intelligent platform
*   [JumpServer](https://github.com/jumpserver/jumpserver/) - Open-source bastion host
*   [Cordys CRM](https://github.com/1Panel-dev/CordysCRM) - Open-source AI CRM
*   [Halo](https://github.com/halo-dev/halo/) - Open-source website builder
*   [MeterSphere](https://github.com/metersphere/metersphere/) - Open-source continuous testing tool

## License

SQLBot is licensed under the [FIT2CLOUD Open Source License](LICENSE), which is essentially GPLv3 with additional limitations.

You can develop derivative works based on SQLBot's source code, but you must adhere to these conditions:

*   Do not replace or modify the SQLBot logo and copyright information.
*   Derivative works must comply with the GPL V3 open-source obligations.

For commercial licensing inquiries, please contact support@fit2cloud.com.
```
Key improvements and SEO considerations:

*   **Clear, Concise Title:** "SQLBot: Your AI-Powered Assistant for Data Insights" is more descriptive.
*   **Strong Hook:** The introductory sentence immediately establishes the value proposition.
*   **Keyword Optimization:** Uses relevant keywords like "AI-powered," "text-to-SQL," "data analysis," and "open-source."
*   **Well-Organized Headings and Structure:** Makes the README easy to scan and understand.
*   **Bulleted Key Features:**  Highlights the core benefits in a clear and concise manner.
*   **Call to Action:** Encourages users to check out the original repo.
*   **Docker Commands Formatting:**  Kept the Docker command and formatted it better.
*   **Complete Information:** Included all essential details from the original README.
*   **SEO Friendly:** Added relevant keywords and phrases throughout the document to improve search engine visibility.