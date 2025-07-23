<div align="center">
    <a href="https://frappe.io/erpnext">
        <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Business Growth</h2>
    <p align="center">
        <p>Manage every aspect of your business with ERPNext, a powerful and intuitive open-source ERP system.</p>
    </p>

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)<br><br>
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

</div>

<div align="center">
    <img src="./erpnext/public/images/v16/hero_image.png"/>
</div>

<div align="center">
    <a href="https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo">Live Demo</a>
    -
    <a href="https://frappe.io/erpnext">Website</a>
    -
    <a href="https://docs.frappe.io/erpnext/">Documentation</a>
    -
    <a href="https://github.com/frappe/erpnext">View on GitHub</a>
</div>

## About ERPNext

ERPNext is a 100% open-source Enterprise Resource Planning (ERP) system designed to streamline and automate your business operations. From accounting to manufacturing, ERPNext offers a comprehensive suite of tools, all available for free.

### Key Features of ERPNext

*   **Accounting:** Comprehensive financial management tools, including transaction recording, reporting, and analysis.
*   **Order Management:** Efficiently track inventory, manage sales orders, and handle customer and supplier interactions.
*   **Manufacturing:** Simplify production cycles, manage material consumption, and optimize capacity planning.
*   **Asset Management:** Track assets from purchase to disposal, ensuring efficient management of your organization's resources.
*   **Projects:** Manage both internal and external projects effectively, with tools for task management, time tracking, and profitability analysis.

<details open>
    <summary>More</summary>
    <img src="https://erpnext.com/files/v16_bom.png"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png"/>
    <img src="https://erpnext.com/files/v16_job_card.png"/>
    <img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Under the Hood

ERPNext is built on the following technologies:

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework (Python/Javascript) providing a robust foundation for building ERPNext, with database abstraction, user authentication, and a REST API.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library which provides a modern user interface.

## Installation and Setup

### Managed Hosting (Recommended)

For ease of use and hassle-free management, consider [Frappe Cloud](https://frappecloud.com), an open-source platform that handles all the complexities of hosting, maintenance, and upgrades.

<div>
    <a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Self-Hosted

#### Docker

1.  **Prerequisites:** Ensure Docker, Docker Compose, and Git are installed.
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
3.  **Run Docker Compose:**
    ```bash
    docker compose -f pwd.yml up -d
    ```
4.  **Access the site:** After a few minutes, your site should be accessible on `localhost:8080`.
    *   Use the default credentials:
        *   **Username:** Administrator
        *   **Password:** admin
    *   See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

#### Manual Install

1.  **Setup bench:** Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation).
2.  **Start the server:** `bench start`
3.  **Create a new site:** `bench new-site erpnext.localhost`
4.  **Get and install ERPNext:**
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
5.  **Access the app:** Open `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn ERPNext and Frappe Framework.
*   [Official documentation](https://docs.erpnext.com/)
*   [Discussion Forum](https://discuss.erpnext.com/)
*   [Telegram Group](https://erpnext_public.t.me)

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Review our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

<br />
<br />
<div align="center" style="padding-top: 0.75rem;">
    <a href="https://frappe.io" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png">
            <img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies" height="28"/>
        </picture>
    </a>
</div>
```
Key improvements and SEO optimizations:

*   **Concise Hook:** "Manage every aspect of your business with ERPNext, a powerful and intuitive open-source ERP system." This is placed right at the beginning to grab the user's attention.
*   **Targeted Keywords:** The title includes the keyword "Open-Source ERP" which is beneficial for SEO. Also utilizes the term "ERP" frequently throughout.
*   **Clear Headings:** Uses clear and descriptive headings (e.g., "About ERPNext," "Key Features of ERPNext," "Installation and Setup") for better readability and SEO ranking.
*   **Bulleted Lists:** Employs bulleted lists for key features, which is beneficial for users and search engine crawlers.
*   **Internal Links:** Added an "About" section to further provide SEO value.  Added the original repo link, and used descriptive anchor text such as "View on GitHub".
*   **Structured Content:** Organizes the README into logical sections, making it easier to understand and navigate.
*   **Call to Action:** Encourages users to try the demo or visit the website.
*   **Focus on Value:** Highlights the benefits of ERPNext clearly, emphasizing its open-source nature and comprehensive features.
*   **Clarity and Readability:** Improves the overall writing style for clarity and conciseness.
*   **Includes links back to the original repo.**