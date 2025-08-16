<!-- Improved README for ERPNext -->

<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Business Growth</h2>
    <p align="center">
        <p>Empower your business with ERPNext, a powerful, intuitive, and 100% open-source ERP system.</p>
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

---

## ERPNext: Your All-in-One Business Solution

**ERPNext** is a comprehensive, open-source Enterprise Resource Planning (ERP) system designed to streamline and automate various business processes. It provides a centralized platform to manage all aspects of your business, from accounting and inventory to manufacturing and customer relationship management.

[Visit the original repository on GitHub](https://github.com/frappe/erpnext)

### Key Features:

*   **Accounting:**  Comprehensive financial management tools to handle transactions, generate reports, and monitor cash flow.
*   **Order Management:** Efficiently manage sales orders, track inventory, and fulfill orders, ensuring timely delivery and customer satisfaction.
*   **Manufacturing:** Simplify production cycles, track material consumption, and optimize capacity planning.
*   **Asset Management:**  Manage assets throughout their lifecycle, from purchase to disposal, ensuring efficient utilization.
*   **Projects:**  Manage projects, track tasks, timesheets, and issues to ensure projects are completed on time and within budget.

<details open>
<summary>More Features</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

---

## Under the Hood:

*   **Frappe Framework:** The robust, full-stack web application framework that powers ERPNext, written in Python and Javascript.  Provides the foundation for secure and scalable applications.
*   **Frappe UI:**  A modern, Vue-based UI library, offering a user-friendly interface for seamless navigation and interaction.

---

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.  It handles installations, upgrades, monitoring, and support, offering peace of mind.

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

**Prerequisites:** docker, docker-compose, git. Refer to [Docker Documentation](https://docs.docker.com) for more details.

**Steps:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your site on `localhost:8080` using the default credentials:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

---

## Development Setup

### Manual Install

The Easy Way: Use the install script for bench to install all dependencies.  See [Bench Documentation](https://github.com/frappe/bench) for details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

**Steps:**

1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to setup bench and start the server.
    ```bash
    bench start
    ```

2.  In a separate terminal:
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

4.  Open `http://erpnext.localhost:8000/app` in your browser.

---

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext.
*   [Official documentation](https://docs.erpnext.com/) - Extensive documentation for ERPNext.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from users.

---

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

---

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