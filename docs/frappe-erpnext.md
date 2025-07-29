<!-- ERPNext: Powerful Open-Source ERP for Your Business -->

<div align="center">
    <a href="https://frappe.io/erpnext">
        <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext</h2>
    <p align="center">
        <b>Supercharge your business with ERPNext, a comprehensive and open-source ERP solution.</b>
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
</div>

## What is ERPNext?

ERPNext is a 100% open-source Enterprise Resource Planning (ERP) system designed to streamline and automate business operations.  Manage all aspects of your business from accounting to manufacturing with this powerful, intuitive platform.

[View the original repository on GitHub](https://github.com/frappe/erpnext)

## Key Features

*   **Accounting:** Comprehensive tools for managing cash flow, from transaction recording to financial reporting and analysis.
*   **Order Management:** Track inventory, manage sales orders, suppliers, shipments, and order fulfillment efficiently.
*   **Manufacturing:** Simplify production cycles, track material consumption, and manage capacity planning.
*   **Asset Management:** Manage assets throughout their lifecycle, from purchase to disposal, across your organization.
*   **Projects:** Manage both internal and external projects, track tasks, timesheets, and issues for improved project delivery and profitability.

<details open>
<summary>More</summary>
    <img src="https://erpnext.com/files/v16_bom.png"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png"/>
    <img src="https://erpnext.com/files/v16_job_card.png"/>
    <img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Under the Hood

ERPNext is built on:

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework written in Python and Javascript.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library for a modern user interface.

## Getting Started

### Production Setup

#### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for hassle-free hosting, taking care of installation, upgrades, monitoring, and support.

<div>
    <a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

#### Self-Hosted

##### Docker

**Prerequisites:** Docker, Docker Compose, Git. See [Docker Documentation](https://docs.docker.com) for setup.

**Instructions:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your site at `localhost:8080`.  Use the following default credentials:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

### Development Setup

#### Manual Install

The Easy Way: Use the install script for bench, which handles dependencies (e.g., MariaDB).  See [https://github.com/frappe/bench](https://github.com/frappe/bench) for more details.

Passwords for the "Administrator" user, MariaDB root user, and the frappe user will be generated and saved to `~/frappe_passwords.txt`.

#### Local Setup

1.  Set up bench (see [Installation Steps](https://frappeframework.com/docs/user/en/installation)) and start the server:
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

1.  [Frappe School](https://school.frappe.io) - Courses on Frappe Framework and ERPNext.
2.  [Official documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from users.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

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