<p align="center"><img src="https://resource-fit2cloud-com.oss-cn-hangzhou.aliyuncs.com/sqlbot/sqlbot.png" alt="SQLBot" width="300" /></p>

# SQLBot: Unlock Data Insights with Conversational AI

SQLBot empowers you to query your data using natural language, making data analysis accessible to everyone. [Explore the SQLBot repository on GitHub](https://github.com/dataease/SQLBot).

<p align="center">
  <a href="https://github.com/dataease/SQLBot/releases/latest"><img src="https://img.shields.io/github/v/release/dataease/SQLBot" alt="Latest release"></a>
  <a href="https://github.com/dataease/SQLBot"><img src="https://img.shields.io/github/stars/dataease/SQLBot?color=%231890FF&style=flat-square" alt="Stars"></a>    
  <a href="https://hub.docker.com/r/dataease/SQLbot"><img src="img.shields.io/docker/pulls/dataease/sqlbot?label=downloads" alt="Download"></a><br/>

</p>

---

## Key Features

*   **Text-to-SQL Conversion:** Seamlessly translate natural language questions into SQL queries using a combination of large language models (LLMs) and Retrieval-Augmented Generation (RAG).
*   **Easy Integration:** Integrate SQLBot into your existing systems or use it with platforms like n8n, MaxKB, Dify, and Coze for enhanced data querying capabilities.
*   **Secure and Controlled:** Implement workspace-based resource isolation and fine-grained data access control for enhanced security and compliance.

## How SQLBot Works

SQLBot leverages advanced AI to understand your questions and translate them into SQL queries, enabling anyone to extract meaningful insights from data.

<img width="1189" height="624" alt="system-arch" src="https://github.com/user-attachments/assets/cde40783-369e-493e-bb59-44ce43c2e7c5" />

## Getting Started

### Installation & Deployment

1.  **Prerequisites:** A Linux server with Docker installed.
2.  **One-Click Installation (Docker):** Run the following command:

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

3.  **Alternative Installation:**
    *   Deploy using [1Panel application store](https://apps.fit2cloud.com/1panel).
    *   For offline environments, use the [offline installation package](https://community.fit2cloud.com/#/products/sqlbot/downloads).

### Accessing SQLBot

*   **Web Interface:** Open your browser and navigate to: `http://<your server IP>:8000/`
*   **Login Credentials:**
    *   Username: `admin`
    *   Password: `SQLBot@123456`

### Contact Us

Join our technical community to discuss any questions you might have.

<img width="180" height="180" alt="contact_me_qr" src="https://github.com/user-attachments/assets/2594ff29-5426-4457-b051-279855610030" />

## UI Demonstration

<img alt="q&a" src="https://github.com/user-attachments/assets/55526514-52f3-4cfe-98ec-08a986259280"   />

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dataease/sqlbot&type=Date)](https://www.star-history.com/#dataease/sqlbot&Date)

## More Projects by Fit2Cloud

*   [DataEase](https://github.com/dataease/dataease/) - Open-source BI tool.
*   [1Panel](https://github.com/1panel-dev/1panel/) - Linux server management panel.
*   [MaxKB](https://github.com/1panel-dev/MaxKB/) - Enterprise-grade intelligent platform.
*   [JumpServer](https://github.com/jumpserver/jumpserver/) - Open-source bastion host.
*   [Cordys CRM](https://github.com/1Panel-dev/CordysCRM) - Open-source AI CRM system.
*   [Halo](https://github.com/halo-dev/halo/) - Open-source website builder.
*   [MeterSphere](https://github.com/metersphere/metersphere/) - Continuous testing tool.

## License

SQLBot is released under the [FIT2CLOUD Open Source License](LICENSE), a license with similar restrictions to GPLv3.

**Usage Guidelines:**

*   You can develop based on the SQLBot source code.
*   Do not modify the SQLBot logo or copyright information.
*   Derivative works must comply with the GPL V3 open-source obligations.

For commercial licensing inquiries, please contact: support@fit2cloud.com.