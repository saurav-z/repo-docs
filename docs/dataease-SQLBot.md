<p align="center"><img src="https://resource-fit2cloud-com.oss-cn-hangzhou.aliyuncs.com/sqlbot/sqlbot.png" alt="SQLBot" width="300" /></p>

# SQLBot: The Intelligent Data Query System

SQLBot is a cutting-edge, AI-powered system that allows users to query data using natural language, making data analysis accessible to everyone.  [See the original repository.](https://github.com/dataease/SQLBot)

[![Latest Release](https://img.shields.io/github/v/release/dataease/SQLBot)](https://github.com/dataease/SQLBot/releases/latest)
[![Stars](https://img.shields.io/github/stars/dataease/SQLBot?color=%231890FF&style=flat-square)](https://github.com/dataease/SQLBot)
[![Docker Pulls](https://img.shields.io/docker/pulls/dataease/sqlbot?label=downloads)](https://hub.docker.com/r/dataease/SQLbot)

## Key Features

*   **Natural Language to SQL:**  Translate your questions into SQL queries effortlessly.
*   **Out-of-the-Box Functionality:** Get started quickly by configuring your large language model (LLM) and data source.
*   **Easy Integration:** Seamlessly integrate SQLBot into your existing systems or platforms like n8n, MaxKB, Dify, and Coze.
*   **Secure and Controllable:** Implement fine-grained data access control with workspace-based resource isolation.

## How it Works

[<img width="1189" height="624" alt="system-arch" src="https://github.com/user-attachments/assets/cde40783-369e-493e-bb59-44ce43c2e7c5" />](https://github.com/dataease/SQLBot)

## Getting Started

### Installation

1.  **Prerequisites:** A Linux server with Docker installed.

2.  **Deployment:** Execute the following one-click installation script:

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

    Alternatively, deploy SQLBot via the [1Panel App Store](https://apps.fit2cloud.com/1panel).

3.  **Offline Installation:** For internal network environments, use the [offline installation package](https://community.fit2cloud.com/#/products/sqlbot/downloads).

### Accessing SQLBot

*   **Web Interface:** Open `http://<Your Server IP>:8000/` in your browser.
*   **Login Credentials:**
    *   Username: `admin`
    *   Password: `SQLBot@123456`

### UI Showcase

  <tr>
    <img alt="q&a" src="https://github.com/user-attachments/assets/55526514-52f3-4cfe-98ec-08a986259280"   />
  </tr>

## Contact Us

For any questions or support, join our technical exchange group using the QR code below:

<img width="180" height="180" alt="contact_me_qr" src="https://github.com/user-attachments/assets/2594ff29-5426-4457-b051-279855610030" />

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dataease/sqlbot&type=Date)](https://www.star-history.com/#dataease/sqlbot&Date)

## Related Projects from FIT2CLOUD

*   [DataEase](https://github.com/dataease/dataease/) - Open-source BI tool
*   [1Panel](https://github.com/1panel-dev/1panel/) - Linux server management panel
*   [MaxKB](https://github.com/1panel-dev/MaxKB/) - Enterprise-grade intelligent platform
*   [JumpServer](https://github.com/jumpserver/jumpserver/) - Open-source bastion host
*   [Halo](https://github.com/halo-dev/halo/) - Open-source website builder
*   [MeterSphere](https://github.com/metersphere/metersphere/) - Open-source continuous testing tool

## License

SQLBot is licensed under the [FIT2CLOUD Open Source License](LICENSE), which is based on GPLv3 but has additional restrictions.

You can develop based on SQLBot's source code, but you need to comply with the following regulations:

*   Do not replace or modify SQLBot's logo and copyright information.
*   Derived works must comply with the GPL V3 open source obligations.

For commercial licensing, please contact support@fit2cloud.com.