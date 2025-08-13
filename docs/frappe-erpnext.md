# ERPNext: Open-Source ERP Software for Your Business

**Streamline your business operations and boost productivity with ERPNext, a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system.**  [View the original repository](https://github.com/frappe/erpnext).

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

<img src="./erpnext/public/images/v16/hero_image.png" alt="ERPNext Screenshot"/>

**[Live Demo](https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo) | [Website](https://frappe.io/erpnext) | [Documentation](https://docs.frappe.io/erpnext/)**

## Key Features of ERPNext

ERPNext offers a comprehensive suite of modules to manage all aspects of your business:

*   **Accounting:** Manage finances with tools for transactions, financial reports, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales orders, and handle order fulfillment.
*   **Manufacturing:** Simplify production cycles, track material consumption, and manage subcontracting.
*   **Asset Management:** Track assets from purchase to disposal, covering infrastructure and equipment.
*   **Projects:** Manage both internal and external projects, tracking tasks, timesheets, and profitability.

<details open>
<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png" alt="BOM"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary"/>
	<img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card"/>
	<img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks"/>
</details>

## Under the Hood

*   **Frappe Framework:**  A full-stack web application framework built on Python and JavaScript, providing a robust foundation. [Frappe Framework GitHub](https://github.com/frappe/frappe)
*   **Frappe UI:**  A Vue-based UI library offering a modern user interface.  [Frappe UI GitHub](https://github.com/frappe/frappe-ui)

## Production Setup

### Managed Hosting

[Frappe Cloud](https://frappecloud.com) offers a simple, user-friendly platform to host Frappe applications, handling installation, upgrades, monitoring, and support.

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

**Prerequisites:** Docker, Docker Compose, Git.

**Instructions:**

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run the Docker Compose command:

    ```bash
    docker compose -f pwd.yml up -d
    ```

    After a few minutes, your site should be accessible on `localhost:8080`. Use the following default credentials:

    *   **Username:** Administrator
    *   **Password:** admin

    See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

For a simpler setup, use the install script: `bench`, which installs dependencies such as MariaDB. More details can be found [here](https://github.com/frappe/bench). New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user, which will be saved to `~/frappe_passwords.txt`.

### Local

1.  Set up bench and start the server by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and running:

    ```bash
    bench start
    ```
2.  In a separate terminal:

    ```bash
    bench new-site erpnext.localhost
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```

3.  Open `http://erpnext.localhost:8000/app` in your browser to see the running app.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn from courses by maintainers and the community.
*   [Official documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from users.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Please review our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

<div align="center" style="padding-top: 0.75rem;">
	<a href="https://frappe.io" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png">
			<img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies" height="28"/>
		</picture>
	</a>
</div>