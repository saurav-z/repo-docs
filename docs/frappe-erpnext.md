# ERPNext: Open-Source ERP for Business Growth

ERPNext is a powerful, intuitive, and open-source Enterprise Resource Planning (ERP) system designed to streamline your business operations and drive growth. ([View on GitHub](https://github.com/frappe/erpnext))

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

<div align="center">
	<img src="./erpnext/public/images/v16/hero_image.png"/>
</div>

[Live Demo](https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo)
- [Website](https://frappe.io/erpnext)
- [Documentation](https://docs.frappe.io/erpnext/)

## Key Features

ERPNext offers a comprehensive suite of modules to manage various aspects of your business:

*   **Accounting:** Manage your finances, track cash flow, and generate financial reports.
*   **Order Management:** Oversee inventory, manage sales orders, and fulfill customer needs efficiently.
*   **Manufacturing:** Simplify the production cycle, track material consumption, and optimize capacity planning.
*   **Asset Management:** Track assets from purchase to disposal, covering all aspects of your organization's infrastructure.
*   **Projects:** Manage both internal and external projects, track tasks, timesheets, and issues for optimal delivery.

<details open>
<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Under the Hood

ERPNext is built upon the following technologies:

*   **Frappe Framework:** A full-stack web application framework providing a robust foundation for the ERP system.  ([Frappe Framework](https://github.com/frappe/frappe))
*   **Frappe UI:** A Vue-based UI library providing a modern and responsive user interface. ([Frappe UI](https://github.com/frappe/frappe-ui))

## Production Setup

### Managed Hosting

Simplify your ERPNext deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.

<div align="center">
	<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

### Self-Hosted

#### Docker

Easily deploy ERPNext using Docker.  Ensure you have Docker, Docker Compose, and Git installed.

Run these commands:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your site at `localhost:8080` (default credentials: Administrator/admin).

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setups.

## Development Setup

### Manual Install

Install all dependencies with bench, then start the server:
```bash
bench start
```

#### Local

Follow these steps to set up the repository locally:

1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up bench. Then start the server:
    ```bash
    bench start
    ```

2.  In a new terminal window, create a site:
    ```bash
    bench new-site erpnext.localhost
    ```

3.  Get and install the ERPNext app:
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```

4.  Open `http://erpnext.localhost:8000/app` in your browser to view the running app.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn from courses by maintainers and the community.
*   [Official documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant support from other users.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Review the [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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