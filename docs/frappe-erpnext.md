<div align="center">
    <a href="https://frappe.io/erpnext">
        <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Your Business</h2>
    <p>
      <b>Empower your business with ERPNext, a powerful, intuitive, and open-source Enterprise Resource Planning (ERP) system.</b>
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
    -
    <a href="https://github.com/frappe/erpnext">View on GitHub</a>
</div>

## What is ERPNext?

ERPNext is a 100% open-source ERP system designed to streamline and automate your business operations. Manage everything from accounting and inventory to manufacturing and project management, all within a single, integrated platform.

## Key Features

*   **Accounting:** Comprehensive tools to manage your finances, including transaction recording, financial reporting, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales orders, handle customer and supplier relationships, and streamline order fulfillment.
*   **Manufacturing:** Simplify the production cycle with features like material consumption tracking, capacity planning, and subcontracting management.
*   **Asset Management:** Track your organization's assets from purchase to disposal, covering IT infrastructure and equipment.
*   **Projects:** Manage both internal and external projects efficiently, tracking tasks, timesheets, and issues to ensure on-time, on-budget, and profitable project delivery.

<details open>
    <summary>More</summary>
    <img src="https://erpnext.com/files/v16_bom.png" alt="Bill of Materials Example" />
    <img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary Example" />
    <img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card Example" />
    <img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks Example" />
</details>

## Technologies Behind ERPNext

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework built on Python and JavaScript, providing the robust foundation for ERPNext.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A modern, Vue-based UI library that provides a user-friendly interface.

## Production Setup

### Managed Hosting

For a hassle-free experience, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. Frappe Cloud handles installation, upgrades, monitoring, and support.

<div>
    <a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png" alt="Try on Frappe Cloud">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Self-Hosted

#### Docker

**Prerequisites:** Docker, Docker Compose, and Git.  Refer to the [Docker Documentation](https://docs.docker.com) for details on setup.

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the Docker Compose command:
    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your ERPNext site should be accessible on `localhost:8080`. Use the following default credentials:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The Easy Way: Use the bench install script, which installs dependencies (e.g., MariaDB). See [Bench Documentation](https://github.com/frappe/bench) for details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
    ```bash
    bench start
    ```

2.  In a separate terminal window, run these commands:
    ```bash
    # Create a new site
    bench new-site erpnext.localhost
    ```

3.  Get and install the ERPNext app:
    ```bash
    # Get the ERPNext app
    bench get-app https://github.com/frappe/erpnext

    # Install the app
    bench --site erpnext.localhost install-app erpnext
    ```

4.  Open the URL `http://erpnext.localhost:8000/app` in your browser. You should see the app running.

## Learning and Community

1.  [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext from the courses by maintainers and the community.
2.  [Official Documentation](https://docs.erpnext.com/) - Comprehensive documentation for ERPNext.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from a large user community.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

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