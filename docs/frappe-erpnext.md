<!-- Improved README for ERPNext -->

<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Growing Businesses</h2>
</div>

<div align="center">
    <p>Empower your business with ERPNext, a powerful and intuitive open-source ERP system. Manage all aspects of your operations, from accounting to manufacturing, with ease.</p>
</div>

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)
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
    -  [View on GitHub](https://github.com/frappe/erpnext)
</div>

## ERPNext: Run Your Business Better with Open-Source ERP

ERPNext is a 100% open-source Enterprise Resource Planning (ERP) system designed to help businesses of all sizes streamline operations, improve efficiency, and drive growth. From accounting and inventory management to manufacturing and project management, ERPNext provides a comprehensive suite of features to manage your entire business.

### Key Features

*   **Accounting:** Manage your finances with a complete suite of tools, including invoicing, expense tracking, financial reporting, and more.
*   **Order Management:**  Track inventory, manage sales orders, handle customer and supplier interactions, and oversee order fulfillment.
*   **Manufacturing:** Simplify your production cycle with features for bill of materials (BOM), production planning, material consumption tracking, and subcontracting.
*   **Asset Management:**  Track your organization's assets from purchase to disposal, including IT infrastructure and equipment.
*   **Projects:**  Deliver projects on time and within budget by tracking tasks, managing timesheets, and monitoring issues.

<details open>
<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Technology

*   **Frappe Framework:** A robust, full-stack web application framework, providing the foundation for ERPNext. Built with Python and JavaScript, it offers a database abstraction layer, user authentication, and a REST API.
*   **Frappe UI:** A Vue.js-based UI library that provides a modern and responsive user interface for a seamless user experience.

## Getting Started

### Managed Hosting (Recommended)

For a hassle-free experience, try [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.  It handles installation, upgrades, monitoring, and support, allowing you to focus on your business.

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

**Prerequisites:** Docker, docker-compose, and Git.

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the Docker Compose file:

    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your site will be accessible on your localhost at port 8080. Use the following default credentials to log in:

*   **Username:** Administrator
*   **Password:** admin

For ARM-based Docker setups, refer to [Frappe Docker documentation](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Install

The easiest way to install all dependencies (e.g., MariaDB) is using our install script for bench.  See [bench documentation](https://github.com/frappe/bench) for more details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:

    ```bash
    bench start
    ```
2.  In a separate terminal window, run:

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
4.  Open the URL `http://erpnext.localhost:8000/app` in your browser to access the running app.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext through courses.
*   [Official documentation](https://docs.erpnext.com/) - Extensive documentation for ERPNext.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from the user community.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## License

ERPNext is licensed under the [MIT License](LICENSE).

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
Key improvements and explanations:

*   **SEO Optimization:** Includes keywords like "open-source ERP," "ERP system," and business process terms like "accounting," "manufacturing," etc.  The title tag (implicitly) is optimized.
*   **Clear Hook:** The one-sentence hook provides an immediate benefit: "Empower your business with ERPNext, a powerful and intuitive open-source ERP system."
*   **Concise and Informative Sections:**  Each section has a clear purpose and uses concise language.
*   **Bulleted Key Features:** Easy to scan and highlights the core value proposition.
*   **Calls to Action:**  Links to live demo, website, and documentation are prominent.  Added a link to the GitHub repo.
*   **Improved Structure:** More logical flow, with clear headings and subheadings for readability.
*   **More Detailed Information:** Expanded on some sections, like the technology section explaining Frappe and Frappe UI.
*   **Contribution Section:** Added links to contribution guidelines and a security vulnerability reporting policy.
*   **License Information:** Clearly states the license (good for open-source projects).
*   **Emphasis on Benefits:** Focuses on *what* ERPNext *does* for the user (e.g., "Streamline operations," "Improve efficiency," "Drive growth")
*   **Removed Redundancy:** Streamlined text, removed unnecessary phrasing.
*   **Markdown Formatting:** Uses clean and consistent Markdown for readability.
*   **Clear instructions:** Includes the instructions in the correct places with appropriate section headers.