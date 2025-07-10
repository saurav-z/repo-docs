<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Business Management</h2>
</div>

<p align="center">
    <b>Empower your business with ERPNext, a powerful, intuitive, and open-source Enterprise Resource Planning (ERP) system.</b>
</p>

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)<br><br>
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

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

---

## What is ERPNext?

ERPNext is a 100% open-source ERP system designed to streamline and automate your business operations. Manage all aspects of your business with this all-in-one solution.

## Key Features

*   **Accounting:** Comprehensive tools for managing your finances, from transactions to financial reporting.
*   **Order Management:** Track inventory, manage sales orders, suppliers, shipments, and fulfillment.
*   **Manufacturing:** Simplify the production cycle, including material tracking, capacity planning, and subcontracting.
*   **Asset Management:** Track assets from purchase to disposal, covering IT infrastructure and equipment.
*   **Projects:** Manage both internal and external projects, tracking tasks, timesheets, and profitability.

<details open>
    <summary>More</summary>
    <img src="https://erpnext.com/files/v16_bom.png" alt="Bill of Materials">
    <img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary">
    <img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card">
    <img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks">
</details>

## Technology

*   **Frappe Framework:** A full-stack web application framework (Python & Javascript) that provides the foundation for ERPNext.  [Learn More](https://github.com/frappe/frappe)
*   **Frappe UI:** A modern Vue-based UI library that provides a user-friendly interface for the application. [Learn More](https://github.com/frappe/frappe-ui)

## Getting Started

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for a simplified and managed hosting experience.  It handles installation, upgrades, and maintenance.

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

**Prerequisites:** Docker, Docker Compose, Git.  See [Docker Documentation](https://docs.docker.com) for setup instructions.

**Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run using Docker Compose:
    ```bash
    docker compose -f pwd.yml up -d
    ```
3.  Access your site at `localhost:8080`. Use the default login credentials:
    *   Username: `Administrator`
    *   Password: `admin`

    See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

### Development Setup

#### Manual Install
Use the `bench` install script as the easy way to set up dependencies.

#### Local

1.  Set up `bench`: Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
    ```bash
    bench start
    ```
2.  In a separate terminal:
    ```bash
    bench new-site erpnext.localhost
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
3.  Open `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community

1.  [Frappe School](https://school.frappe.io) - Learn from courses by maintainers and the community.
2.  [Official Documentation](https://docs.erpnext.com/) - Comprehensive documentation.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with the community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Please review our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

---

<div align="center" style="padding-top: 0.75rem;">
    <a href="https://frappe.io" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png">
            <img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies" height="28"/>
        </picture>
    </a>
</div>