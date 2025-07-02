<!-- Improved & SEO-Optimized README -->

<div align="center">
  <a href="https://frappe.io/erpnext">
    <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
  </a>
  <h1>ERPNext: Open-Source ERP Software for Business Management</h1>
  <p><b>Manage your entire business with ERPNext, a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system.</b></p>

  [![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)<br><br>
  [![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
  [![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)
</div>

<div align="center">
  <img src="./erpnext/public/images/v16/hero_image.png" alt="ERPNext Hero Image"/>
</div>

<div align="center">
  <a href="https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo">Live Demo</a>
  - <a href="https://frappe.io/erpnext">Website</a>
  - <a href="https://docs.frappe.io/erpnext/">Documentation</a>
</div>

## What is ERPNext?

ERPNext is a comprehensive, open-source ERP solution designed to streamline business operations across various departments.  It's a complete ERP, offering modules for everything from accounting to manufacturing, all in one integrated system.  **[View the source code on GitHub](https://github.com/frappe/erpnext)**.

## Key Features of ERPNext

ERPNext offers a wide range of features to manage your business effectively:

*   ✅ **Accounting:** Manage your finances, track cash flow, generate financial reports, and handle all your accounting needs.
*   ✅ **Order Management:**  Handle sales orders, manage inventory, track stock levels, manage customers and suppliers.
*   ✅ **Manufacturing:** Optimize the production cycle, manage material consumption, and plan capacity.
*   ✅ **Asset Management:**  Track your organization's assets from purchase to disposal, including IT infrastructure and equipment.
*   ✅ **Projects:** Manage projects, track tasks, timesheets, and issues for better project delivery and profitability.
<details open>
<summary>More</summary>
  <img src="https://erpnext.com/files/v16_bom.png" alt="BOM"/>
  <img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary"/>
  <img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card"/>
  <img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks"/>
</details>

## Under the Hood: Technologies

*   **Frappe Framework:** A full-stack web application framework, written in Python and Javascript, providing the foundation for ERPNext. ( [Frappe Framework](https://github.com/frappe/frappe) )
*   **Frappe UI:** A Vue.js-based UI library to provide a modern and user-friendly interface.  ( [Frappe UI](https://github.com/frappe/frappe-ui) )

## Deployment Options

### Managed Hosting

For ease of use, try [Frappe Cloud](https://frappecloud.com), a user-friendly open-source platform that handles setup, upgrades, monitoring, and maintenance.

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

**Prerequisites:** docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

**Steps:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your site on `localhost:8080` after a few minutes.  Use the default credentials:
*   Username: Administrator
*   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

For a simpler setup, use our install script for bench. See [bench](https://github.com/frappe/bench) for details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

1.  **Set up bench:** Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server.
    ```bash
    bench start
    ```
2.  **Create a new site:** In a separate terminal.
    ```bash
    bench new-site erpnext.localhost
    ```
3.  **Get and install ERPNext app:**
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
4.  **Access your app:** Open `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community Resources

*   [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext.
*   [Official documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from the user community.

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