# ERPNext: The Open-Source ERP That Empowers Your Business

**Unlock the power of streamlined operations with ERPNext, a fully open-source ERP system designed to help you manage every aspect of your business, from accounting to manufacturing.** ([Original Repository](https://github.com/frappe/erpnext))

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)<br><br>
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

[<img src="./erpnext/public/images/v16/hero_image.png" alt="ERPNext Dashboard" width="100%"/>](https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo)

[Live Demo](https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo) | [Website](https://frappe.io/erpnext) | [Documentation](https://docs.frappe.io/erpnext/)

## Key Features of ERPNext

ERPNext offers a comprehensive suite of features to manage and optimize your business processes:

*   **Accounting:** Streamline your finances with robust tools for managing cash flow, recording transactions, and generating insightful financial reports.
*   **Order Management:** Efficiently track inventory, manage sales orders, handle customer and supplier relationships, and optimize order fulfillment.
*   **Manufacturing:** Simplify your production cycle with tools for tracking material consumption, capacity planning, subcontracting, and more.
*   **Asset Management:**  Track and manage your assets, from purchase to disposal, across all departments.
*   **Projects:**  Manage both internal and external projects, ensuring they stay on time, within budget, and profitable. Track tasks, timesheets, and issues.

<details open>
<summary>More Features</summary>
    <img src="https://erpnext.com/files/v16_bom.png" alt="Bill of Materials" width="30%"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary" width="30%"/>
    <img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card" width="30%"/>
    <img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks" width="30%"/>
</details>

## Technology Stack

ERPNext is built upon the following technologies:

*   **Frappe Framework:** A robust, full-stack web application framework written in Python and Javascript, providing the foundation for ERPNext. ([Frappe Framework](https://github.com/frappe/frappe))
*   **Frappe UI:** A Vue-based UI library that provides a modern user interface. ([Frappe UI](https://github.com/frappe/frappe-ui))

## Deployment Options

Choose the deployment option that best fits your needs:

### Managed Hosting

Simplify deployment and maintenance with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.  It handles installation, upgrades, monitoring, and support.

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

**Prerequisites:** docker, docker-compose, git.

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the Docker Compose file:
    ```bash
    docker compose -f pwd.yml up -d
    ```

Access your site on `localhost:8080` using the default credentials:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based docker setup.

## Development Setup

### Manual Install

Refer to [Installation Steps](https://frappeframework.com/docs/user/en/installation) for detailed instructions.

### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:

    ```bash
    bench start
    ```
2.  In a separate terminal:

    ```bash
    # Create a new site
    bench new-site erpnext.localhost
    # Get the ERPNext app
    bench get-app https://github.com/frappe/erpnext
    # Install the app
    bench --site erpnext.localhost install-app erpnext
    ```

3.  Open `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn ERPNext and Frappe Framework.
*   [Official documentation](https://docs.erpnext.com/) - Comprehensive documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

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