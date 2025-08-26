<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
</div>

## ERPNext: Open-Source ERP for Business Management

**Manage your entire business with ERPNext, a powerful and open-source Enterprise Resource Planning (ERP) system, empowering you to streamline operations and boost efficiency.** [Explore the original repository](https://github.com/frappe/erpnext).

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
</div>

### Key Features

*   **Accounting:** Comprehensive tools for managing finances, from transactions to financial reports.
*   **Order Management:** Track inventory, manage sales orders, suppliers, shipments, and fulfillment.
*   **Manufacturing:** Streamline production cycles, manage material consumption, and plan capacity.
*   **Asset Management:** Track assets from purchase to disposal, across your entire organization.
*   **Project Management:** Manage projects, track tasks, timesheets, and issues for on-time and within-budget delivery.

<details open>
    <summary>More Features</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Core Technologies

*   **Frappe Framework:** The robust Python and JavaScript-based full-stack web application framework underlying ERPNext. ([Frappe Framework GitHub](https://github.com/frappe/frappe))
*   **Frappe UI:** A modern, Vue.js-based UI library providing a user-friendly interface. ([Frappe UI GitHub](https://github.com/frappe/frappe-ui))

### Production Setup

#### Managed Hosting

Simplify your ERPNext deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting and managing Frappe applications.

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

**Prerequisites:** docker, docker-compose, git.

**Instructions:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your site on `localhost:8080` using default credentials:

*   Username: Administrator
*   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

### Development Setup

#### Manual Install

The easiest way is with our install script. See [bench](https://github.com/frappe/bench) for details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

#### Local

To set up the repository locally:

1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
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

### Learning and Community

1.  [Frappe School](https://school.frappe.io) - Learn ERPNext and the Frappe Framework.
2.  [Official Documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from other users.

### Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

### Logo and Trademark Policy

Please review the [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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