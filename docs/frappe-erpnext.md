<div align="center">
    <a href="https://frappe.io/erpnext">
	    <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
</div>

# ERPNext: Open-Source ERP for Business Management

**ERPNext is a powerful, intuitive, and open-source Enterprise Resource Planning (ERP) system designed to streamline your business operations.**

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

<div align="center">
	<img src="./erpnext/public/images/v16/hero_image.png"/>
</div>

[Live Demo](https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo)
-
[Website](https://frappe.io/erpnext)
-
[Documentation](https://docs.frappe.io/erpnext/)
-
[**View the original repository on GitHub**](https://github.com/frappe/erpnext)

## Key Features of ERPNext

ERPNext offers a comprehensive suite of modules to manage your business effectively. Key features include:

*   **Accounting:** Manage your finances with ease, from transaction recording to financial reporting.
*   **Order Management:** Track inventory, manage sales orders, and fulfill customer needs.
*   **Manufacturing:** Simplify production cycles, track material consumption, and manage subcontracting.
*   **Asset Management:** Track assets from purchase to disposal, covering IT infrastructure and equipment.
*   **Projects:** Manage both internal and external projects, ensuring on-time, on-budget delivery.

<details open>
    <summary>More</summary>
        <img src="https://erpnext.com/files/v16_bom.png"/>
        <img src="https://erpnext.com/files/v16_stock_summary.png"/>
        <img src="https://erpnext.com/files/v16_job_card.png"/>
        <img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Under the Hood

ERPNext is built upon robust technologies:

*   **Frappe Framework:** A full-stack web application framework (Python and JavaScript) providing a solid foundation for web applications.
*   **Frappe UI:** A modern, Vue-based UI library for a responsive and user-friendly interface.

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for a user-friendly, open-source platform for hosting and managing your ERPNext deployments. It handles installation, upgrades, and maintenance.

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

**Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run Docker Compose:
    ```bash
    docker compose -f pwd.yml up -d
    ```
3.  Access your site at `http://localhost:8080`. Use the default login credentials.

#### Manual Install

Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up your bench and start the server.

## Development Setup

### Local Setup

1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to install bench and start the server using `bench start`.
2.  In a separate terminal, run the following:
    ```bash
    bench new-site erpnext.localhost
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
3.  Open `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Refer to our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

<div align="center" style="padding-top: 0.75rem;">
	<a href="https://frappe.io" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png">
			<img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies" height="28"/>
		</picture>
	</a>
</div>