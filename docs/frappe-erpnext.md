<!-- ERPNext README SEO Optimized -->

<div align="center">
    <a href="https://frappe.io/erpnext">
        <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Growing Businesses</h2>
    <p align="center">
        **ERPNext is a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system designed to streamline your business operations.**
    </p>

    <p>
        <a href="https://frappe.school">Learn ERPNext</a>
        <br>
        <br>
        <a href="https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule">
            <img src="https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule" alt="CI Status">
        </a>
        <a href="https://hub.docker.com/r/frappe/erpnext-worker">
            <img src="https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg" alt="Docker Pulls">
        </a>
    </p>
</div>

<div align="center">
    <img src="./erpnext/public/images/v16/hero_image.png" alt="ERPNext Hero Image"/>
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

ERPNext is a comprehensive, **open-source ERP solution** that empowers businesses of all sizes to manage their operations efficiently.  From accounting to manufacturing, ERPNext provides a centralized platform to run your entire business, free of charge.

### Key Features of ERPNext

*   **Accounting:** Manage your finances with tools for transactions, financial reports, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales orders, handle customer relationships, and streamline order fulfillment.
*   **Manufacturing:** Simplify production cycles, track material consumption, manage capacity planning, and handle subcontracting.
*   **Asset Management:** Track assets throughout their lifecycle, from purchase to disposal, covering all branches of your organization.
*   **Projects:** Deliver projects on time and within budget. Track tasks, timesheets, and issues for enhanced project profitability.

<details open>
    <summary>More Screenshots</summary>
        <img src="https://erpnext.com/files/v16_bom.png" alt="BOM Screenshot"/>
        <img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary Screenshot"/>
        <img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card Screenshot"/>
        <img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks Screenshot"/>
</details>

### Technical Underpinnings

*   **Frappe Framework:**  A robust, full-stack web application framework built with Python and Javascript, providing the foundation for ERPNext. ([Frappe Framework on GitHub](https://github.com/frappe/frappe))
*   **Frappe UI:** A modern, Vue-based UI library that provides an intuitive user interface. ([Frappe UI on GitHub](https://github.com/frappe/frappe-ui))

## Installation and Deployment

### Managed Hosting

For a hassle-free experience, consider [Frappe Cloud](https://frappecloud.com). It offers a user-friendly platform for hosting Frappe applications, handling installation, upgrades, monitoring, and maintenance.

<div>
    <a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28"/>
        </picture>
    </a>
</div>

### Self-Hosted

#### Docker

**Prerequisites:**  Docker, Docker Compose, and Git are required. Refer to the [Docker Documentation](https://docs.docker.com) for detailed setup instructions.

**Installation Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Start the containers:
    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, access your ERPNext instance at `localhost:8080`.  Use the following default login credentials:

*   **Username:** Administrator
*   **Password:** admin

Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

#### Manual Installation

For local development or self-hosting with a manual setup:

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
    ```bash
    bench start
    ```
2.  Open a separate terminal and create a new site:
    ```bash
    bench new-site erpnext.localhost
    ```
3.  Get and install the ERPNext app:
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
4.  Access your ERPNext instance at `http://erpnext.localhost:8000/app` in your browser.

## Learning Resources and Community

*   [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext through various courses.
*   [Official Documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from a large user base.

## Contributing to ERPNext

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Trademark Policy

Please review our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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
Key improvements and explanations:

*   **SEO-Friendly Title:**  The title includes the primary keyword ("ERPNext") and the benefit ("Open-Source ERP") and  target audience ("Growing Businesses").
*   **Concise Hook:** The opening sentence grabs attention immediately.
*   **Clear Headings:** Uses consistent and descriptive headings to organize information (e.g., "About ERPNext," "Key Features," "Installation and Deployment," etc.).
*   **Bulleted Lists:**  Uses bullet points for easy readability of key features and steps.
*   **Expanded Key Features:** Added brief descriptions for each feature to improve understanding.
*   **Technical Details Section:** Provides more context on the underlying technologies.
*   **Deployment Options:**  Clearly outlines both managed hosting (Frappe Cloud) and self-hosting (Docker and Manual).
*   **Community Links:**  Emphasizes learning and community engagement.
*   **Contribution Section:**  Directs users to relevant resources for contributing.
*   **Removed Redundancy:** Condensed repetitive phrases.
*   **Added Alt text:** included alt text for images.
*   **Markdown Formatting:** Consistent and clean markdown formatting for improved readability.
*   **GitHub Link:** Added the GitHub link to the About ERPNext section.

This improved README is more informative, user-friendly, and SEO-optimized, making it easier for users to understand and explore ERPNext.