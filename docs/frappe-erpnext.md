<div align="center">
    <a href="https://frappe.io/erpnext">
	    <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Growing Businesses</h2>
    <p align="center">
        <p>Power your business with ERPNext, a powerful, intuitive, and open-source ERP system.</p>
    </p>

    [![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)<br><br>
    [![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
    [![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)
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

## What is ERPNext?

ERPNext is a 100% open-source Enterprise Resource Planning (ERP) system designed to help businesses manage all core operations in one place. From accounting to manufacturing, ERPNext streamlines processes and boosts efficiency.

## Key Features of ERPNext

*   **Accounting:** Manage finances, track cash flow, and generate comprehensive financial reports.
*   **Order Management:** Oversee inventory, handle sales orders, manage customers and suppliers, and ensure order fulfillment.
*   **Manufacturing:** Simplify production cycles, track material consumption, and facilitate capacity planning.
*   **Asset Management:** Track assets from purchase to disposal across your entire organization.
*   **Projects:** Manage both internal and external projects, track tasks, timesheets, and profitability.

<details open>
    <summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png" alt="BOM"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary"/>
	<img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card"/>
	<img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks"/>
</details>

## Technology Stack

*   **[Frappe Framework](https://github.com/frappe/frappe):** A full-stack web application framework built with Python and JavaScript, providing the foundation for ERPNext.
*   **[Frappe UI](https://github.com/frappe/frappe-ui):** A Vue-based UI library that offers a modern and user-friendly interface.

## Getting Started

### Production Setup

*   **Managed Hosting:** Explore [Frappe Cloud](https://frappecloud.com) for hassle-free hosting, with automatic setup, upgrades, and maintenance.

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

1.  **Prerequisites:** Docker, Docker Compose, and Git.
2.  **Run the following commands:**

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    docker compose -f pwd.yml up -d
    ```

    Access the site at `http://localhost:8080`. Use the following default login credentials:
    *   Username: `Administrator`
    *   Password: `admin`
    For ARM-based Docker setups, refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

### Development Setup

#### Manual Install

1.  **Install Bench:** Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up bench and start the server.
    ```bash
    bench start
    ```
2.  **In a separate terminal window:**

    ```bash
    # Create a new site
    bench new-site erpnext.localhost
    # Get the ERPNext app
    bench get-app https://github.com/frappe/erpnext
    # Install the app
    bench --site erpnext.localhost install-app erpnext
    ```
3.  Open the URL `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community Resources

1.  [Frappe School](https://school.frappe.io) - Learn ERPNext and the Frappe Framework.
2.  [Official documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from the user community.

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