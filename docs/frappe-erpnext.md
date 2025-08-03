<div align="center">
    <a href="https://frappe.io/erpnext">
        <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Growing Businesses</h2>
    <p align="center">
        <b>Manage your entire business with a single, powerful, and open-source ERP solution.</b>
    </p>
</div>

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)<br><br>
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

<div align="center">
    <img src="./erpnext/public/images/v16/hero_image.png" alt="ERPNext Hero Image"/>
</div>

<div align="center">
    <a href="https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo">Live Demo</a>
    -
    <a href="https://frappe.io/erpnext">Website</a>
    -
    <a href="https://docs.frappe.io/erpnext/">Documentation</a>
</div>

## What is ERPNext?

ERPNext is a 100% open-source Enterprise Resource Planning (ERP) system designed to streamline your business operations. From accounting and inventory management to manufacturing and project management, ERPNext provides a comprehensive suite of tools to help you manage and grow your business.

**[Explore the ERPNext GitHub Repository](https://github.com/frappe/erpnext)**

### Key Features:

*   **Accounting:** Comprehensive financial management tools, from transactions to financial reporting.
*   **Order Management:** Manage sales, inventory, and fulfillment efficiently.
*   **Manufacturing:** Simplify your production cycle with features like BOM, capacity planning, and subcontracting.
*   **Asset Management:** Track and manage your organization's assets, from purchase to disposal.
*   **Projects:** Deliver projects on time, within budget, and achieve profitability, with features like task management and timesheets.

<details open>
    <summary>More</summary>
    <img src="https://erpnext.com/files/v16_bom.png" alt="BOM"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary"/>
    <img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card"/>
    <img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks"/>
</details>

## Technologies Under the Hood

*   **[Frappe Framework](https://github.com/frappe/frappe):** A powerful, full-stack web application framework built with Python and JavaScript.
*   **[Frappe UI](https://github.com/frappe/frappe-ui):** A Vue.js-based UI library for a modern and user-friendly interface.

## Production Setup

### Managed Hosting

Get started quickly and easily with [Frappe Cloud](https://frappecloud.com), the perfect platform to host Frappe applications with peace of mind. It takes care of installation, setup, upgrades, monitoring, maintenance, and support.

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

**Prerequisites:** Docker, Docker Compose, and Git. Refer to the [Docker Documentation](https://docs.docker.com) for setup instructions.

**Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the Docker Compose command:
    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, access your site via localhost port 8080. Use the following default credentials:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for instructions to run on ARM64 architecture.

## Development Setup

### Manual Install

The Easy Way: Use our install script for bench to install all dependencies (e.g., MariaDB). See [Bench Documentation](https://github.com/frappe/bench) for more details.

The script creates new passwords for:

*   ERPNext "Administrator" user
*   MariaDB root user
*   frappe user (passwords are saved to `~/frappe_passwords.txt`)

### Local

Follow these steps to set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal, create a new site:

    ```bash
    bench new-site erpnext.localhost
    ```

3.  Get and install the ERPNext app:

    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```

4.  Open the URL `http://erpnext.localhost:8000/app` in your browser. You should see the app running.

## Learning and Community

*   [Frappe School](https://school.frappe.io): Learn the Frappe Framework and ERPNext.
*   [Official documentation](https://docs.erpnext.com/): Extensive documentation for ERPNext.
*   [Discussion Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me): Get instant help from other users.

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