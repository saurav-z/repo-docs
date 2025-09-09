<!-- SEO-optimized README for ERPNext -->

<div align="center">
    <a href="https://frappe.io/erpnext">
        <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
</div>

# ERPNext: Open-Source ERP Software for Your Business

**ERPNext is a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system designed to streamline and optimize your business operations.**  ([See the original repo](https://github.com/frappe/erpnext))

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

## Key Features of ERPNext

ERPNext provides a comprehensive suite of tools to manage your business, offering features like:

*   **Accounting:** Manage your finances with tools for transactions, financial reports, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales and purchase orders, and streamline order fulfillment.
*   **Manufacturing:** Simplify production cycles, manage material consumption, and optimize manufacturing processes.
*   **Asset Management:** Track your organization's assets from purchase to disposal, covering all departments.
*   **Projects:** Deliver projects on time and on budget with integrated task and issue tracking.

<details open>
<summary>More</summary>
    <img src="https://erpnext.com/files/v16_bom.png"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png"/>
    <img src="https://erpnext.com/files/v16_job_card.png"/>
    <img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Under the Hood

ERPNext is built on the following powerful technologies:

*   **Frappe Framework:** A full-stack web application framework (Python/JavaScript) that provides a robust foundation for building web applications, including a database abstraction layer, user authentication, and a REST API. ([Frappe Framework](https://github.com/frappe/frappe))
*   **Frappe UI:** A Vue-based UI library that provides a modern user interface for your ERPNext application. ([Frappe UI](https://github.com/frappe/frappe-ui))

## Production Setup

Choose the best hosting solution for your needs:

### Managed Hosting

*   **Frappe Cloud:** A user-friendly platform for hosting Frappe applications. Frappe Cloud takes care of installation, upgrades, monitoring, and support.

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

*   **Prerequisites:** Docker, docker-compose, git.
*   **Setup:**
    1.  Clone the repository: `git clone https://github.com/frappe/frappe_docker`
    2.  Navigate to the directory: `cd frappe_docker`
    3.  Run Docker Compose: `docker compose -f pwd.yml up -d`
*   Your site should be accessible on `localhost:8080` after a few minutes.
*   Default login: Username: `Administrator`, Password: `admin`

#### For ARM based docker setup
*   See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions)

## Development Setup

### Manual Install

The Easy Way: our install script for bench will install all dependencies (e.g. MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

Follow these steps to set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
    ```
    bench start
    ```
2.  In a separate terminal window, run the following commands:
    ```
    # Create a new site
    bench new-site erpnext.localhost
    ```
3.  Get the ERPNext app and install it
    ```
    # Get the ERPNext app
    bench get-app https://github.com/frappe/erpnext

    # Install the app
    bench --site erpnext.localhost install-app erpnext
    ```
4.  Open the URL `http://erpnext.localhost:8000/app` in your browser, you should see the app running

## Learning and Community

Get help and learn more with these resources:

*   [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext.
*   [Official documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from users.

## Contributing

We welcome contributions! Please review:

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

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
```

Key improvements and SEO considerations:

*   **Clear Hook:** The opening sentence clearly states what ERPNext *is* and its key benefit.
*   **Headings and Structure:** Uses headings (H2, H3, etc.) to organize content, making it easier to read and scan.
*   **Keyword Optimization:** Includes keywords like "open-source ERP," "ERP software," and core business functions.
*   **Bulleted Lists:** Uses bulleted lists for key features, making them easy to digest.
*   **Concise Language:** Keeps descriptions brief and to the point.
*   **Links:** Includes relevant links (website, documentation, demo, etc.).  Links to the original repo are also included.
*   **Call to Action (implied):** Encourages users to try the demo, explore documentation, and learn more.
*   **Alt Text:**  Good use of descriptive `alt` text for images.