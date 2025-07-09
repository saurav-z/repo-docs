<div align="center">
    <a href="https://frappe.io/erpnext">
        <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Growing Businesses</h2>
    <p>
        <p>Manage your entire business operations with ERPNext, a powerful, intuitive, and open-source ERP system.</p>
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
    - <a href="https://github.com/frappe/erpnext">View on GitHub</a>
</div>

## What is ERPNext?

ERPNext is a 100% open-source Enterprise Resource Planning (ERP) system designed to help businesses streamline operations and improve efficiency. It provides a comprehensive suite of tools to manage various aspects of your business, from accounting and inventory to manufacturing and project management.

### Key Features

*   **Accounting:** Manage your finances with ease, from transaction recording to financial reporting.
*   **Order Management:** Track inventory, manage sales orders, customers, suppliers, and fulfill orders efficiently.
*   **Manufacturing:** Simplify your production cycle, track material consumption, and manage capacity planning.
*   **Asset Management:** Track and manage your organization's assets, from IT infrastructure to equipment.
*   **Projects:** Manage both internal and external projects on time and within budget, tracking tasks, timesheets, and issues.

<details open>
<summary>More Features</summary>
    <img src="https://erpnext.com/files/v16_bom.png"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png"/>
    <img src="https://erpnext.com/files/v16_job_card.png"/>
    <img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Under the Hood

ERPNext is built on robust and open-source technologies:

*   **Frappe Framework:** A full-stack web application framework (Python and Javascript) providing a solid foundation for building web applications, with database abstraction and REST APIs.
*   **Frappe UI:** A Vue-based UI library providing a modern and user-friendly interface.

## Getting Started with ERPNext

### Managed Hosting

For a hassle-free experience, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. It handles installation, upgrades, monitoring, and support.

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

**Prerequisites:** Docker, docker-compose, git. Refer to the [Docker Documentation](https://docs.docker.com) for setup details.

**Steps:**

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run the Docker Compose file:

    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your site should be accessible on `localhost:8080`. Use the following default credentials:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for instructions on setting up ARM-based Docker.

## Development Setup

### Manual Install

The easiest method is to use the install script for bench, which installs all dependencies. See [bench installation](https://github.com/frappe/bench).

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local

To set up the repository locally, follow these steps:

1.  Set up bench: Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:

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

## Learn and Contribute

### Learning Resources
1.  [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext.
2.  [Official documentation](https://docs.erpnext.com/) - Comprehensive documentation for ERPNext.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with the community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from the community.

### Contributing
1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

## Legal

*   Please review our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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