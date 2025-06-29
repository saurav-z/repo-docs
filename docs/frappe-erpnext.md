<!-- Improved & SEO-Optimized README for ERPNext -->

<div align="center">
    <a href="https://frappe.io/erpnext">
        <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Business Growth</h2>
    <p align="center">
        <b>Transform your business with ERPNext, a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system.</b>
    </p>

    <!-- Badges -->
    <br>
    <a href="https://frappe.school"><img src="https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square" alt="Learn ERPNext on Frappe School"></a>
    <br><br>
    <a href="https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml"><img src="https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule" alt="CI Status"></a>
    <a href="https://hub.docker.com/r/frappe/erpnext-worker"><img src="https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg" alt="Docker Pulls"></a>
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

---

## About ERPNext

ERPNext is a comprehensive, **100% open-source ERP (Enterprise Resource Planning) system** designed to streamline and automate your business operations. From accounting and inventory management to manufacturing and project management, ERPNext offers a complete suite of tools to help you run your business efficiently.

### Key Features & Benefits

ERPNext offers a rich feature set to manage all aspects of your business.

*   ✅ **Accounting:** Manage your finances with tools for transactions, financial reports, and cash flow analysis.
*   ✅ **Order Management:** Track inventory, manage sales orders, handle customer and supplier relationships, and streamline order fulfillment.
*   ✅ **Manufacturing:** Simplify your production cycle with features for bill of materials (BOM), production planning, material consumption tracking, and more.
*   ✅ **Asset Management:** Manage your assets throughout their lifecycle, from purchase to disposal, within a centralized system.
*   ✅ **Project Management:** Deliver projects on time and within budget by tracking tasks, timesheets, and issues.

<details open>
<summary>More</summary>
    <img src="https://erpnext.com/files/v16_bom.png" alt="Bill of Materials Example"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary Example"/>
    <img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card Example"/>
    <img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks Example"/>
</details>

---

## Under the Hood: Technologies

ERPNext is built on a powerful and flexible technology stack.

*   [**Frappe Framework**](https://github.com/frappe/frappe): The full-stack web application framework (Python and JavaScript) that provides the foundation for ERPNext. Features database abstraction, user authentication, and a REST API.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue.js-based UI library that provides a modern and user-friendly interface for ERPNext.

---

## Getting Started: Installation and Setup

### Managed Hosting

For a hassle-free experience, try **Frappe Cloud**, a user-friendly platform for hosting Frappe applications.  It handles installation, upgrades, monitoring, maintenance, and support.

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

1.  **Prerequisites:** Docker, Docker Compose, and Git. Refer to the [Docker documentation](https://docs.docker.com) for setup details.
2.  **Run the following commands:**

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    docker compose -f pwd.yml up -d
    ```

3.  Access your site on `localhost:8080` (after a few minutes).
4.  **Default Login:**  `Username: Administrator`, `Password: admin`
5.  See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

#### Manual Install

Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) on the Frappe Framework documentation. You will need to install bench which will take care of MariaDB.

1.  Start the server: `bench start`
2.  Open a new terminal and run `bench new-site erpnext.localhost`
3.  Install the ERPNext app:
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
4.  Access the app at `http://erpnext.localhost:8000/app`

---

## Resources and Community

*   📚 [Frappe School](https://school.frappe.io) - Learn from the community and the maintainers.
*   📖 [Official Documentation](https://docs.erpnext.com/) - Comprehensive documentation.
*   💬 [Discussion Forum](https://discuss.erpnext.com/) - Connect with the ERPNext community.
*   💬 [Telegram Group](https://erpnext_public.t.me) - Get instant help.

---

## Contribute

We welcome contributions from the community!

*   📝 [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   🛡️ [Report Security Vulnerabilities](https://erpnext.com/security)
*   🤝 [Contribution Guidelines](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   🌐 [Translations](https://crowdin.com/project/frappe)

---

## Legal

*   [Logo and Trademark Policy](TRADEMARK_POLICY.md)

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