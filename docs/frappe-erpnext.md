<div align="center">
    <a href="https://frappe.io/erpnext">
        <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
</div>

# ERPNext: Open-Source ERP Software for Business Management

**ERPNext is a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system designed to help businesses streamline operations and boost efficiency.**  ([View the original repository](https://github.com/frappe/erpnext))

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

## Key Features of ERPNext:

*   **Accounting:** Manage cash flow with comprehensive tools for transactions, financial reports, and analysis.
*   **Order Management:** Track inventory, manage sales orders, suppliers, shipments, and fulfill orders efficiently.
*   **Manufacturing:** Simplify production cycles, monitor material consumption, plan capacity, and manage subcontracting.
*   **Asset Management:** Track assets from purchase to disposal, covering IT infrastructure and equipment.
*   **Projects:** Manage internal and external projects, track tasks, timesheets, and issues for timely delivery.

<details open>
    <summary>More</summary>
    <img src="https://erpnext.com/files/v16_bom.png" alt="BOM"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary"/>
    <img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card"/>
    <img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks"/>
</details>

## Under the Hood

*   **Frappe Framework:** A full-stack web application framework (Python/JavaScript) providing the foundation for ERPNext. ([Frappe Framework](https://github.com/frappe/frappe))
*   **Frappe UI:**  A Vue-based UI library for a modern and responsive user interface. ([Frappe UI](https://github.com/frappe/frappe-ui))

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for hassle-free hosting, with automated setup, upgrades, monitoring, and support.

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

*   **Prerequisites:** docker, docker-compose, git. Refer to [Docker Documentation](https://docs.docker.com) for setup.
*   **Instructions:**

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    docker compose -f pwd.yml up -d
    ```

*   Access the site on `localhost:8080` using the default credentials:
    *   Username: `Administrator`
    *   Password: `admin`
*   See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based setup.

## Development Setup

### Manual Install

*   Use the install script for bench (see [bench](https://github.com/frappe/bench)) to install dependencies.
*   New passwords will be created and saved in `~/frappe_passwords.txt`.

### Local Setup

1.  Install and start bench following the [Installation Steps](https://frappeframework.com/docs/user/en/installation).
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

1.  [Frappe School](https://school.frappe.io) - Learn from courses on Frappe Framework and ERPNext.
2.  [Official documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from users.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Please read our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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