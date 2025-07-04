<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Growing Businesses</h2>
    <p align="center">
        Empower your business with ERPNext, a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system.
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
	-
	<a href="https://github.com/frappe/erpnext">View on GitHub</a>
</div>

## About ERPNext

ERPNext is a comprehensive, **open-source ERP system** designed to streamline and automate your business operations.  From accounting to manufacturing, ERPNext offers a complete suite of tools to manage every aspect of your business.  This open-source nature empowers you with full control and customization, backed by a vibrant community.

### Key Features of ERPNext:

*   **Accounting:** Manage cash flow, record transactions, and generate financial reports.
*   **Order Management:** Track inventory, manage sales orders, and fulfill customer orders efficiently.
*   **Manufacturing:** Simplify production cycles, track material consumption, and optimize manufacturing processes.
*   **Asset Management:** Track assets from purchase to disposal, ensuring efficient resource management.
*   **Projects:** Manage both internal and external projects, track tasks, and monitor profitability.
*   **[View More Features on the official website](https://frappe.io/erpnext)**

<details open>

<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Tech Stack: Under the Hood

ERPNext is built on robust open-source technologies:

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework (Python/Javascript) providing the core infrastructure.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library to provide a modern user interface.

## Getting Started with ERPNext

### Production Setup

Choose the best way to get started for you.

#### Managed Hosting: Frappe Cloud

For simplicity and ease of use, consider [Frappe Cloud](https://frappecloud.com), a fully managed hosting platform. It takes care of installation, updates, monitoring, and support.

<div>
	<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

#### Self-Hosted: Docker

For self-hosting, we recommend using Docker.

**Prerequisites:** Docker, Docker Compose, and Git installed.

**Installation:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a few minutes, your ERPNext instance will be accessible on `http://localhost:8080`.

*   **Login:** Use the default credentials:  Username: `Administrator`, Password: `admin`.

For more details, consult the [Frappe Docker documentation](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

### Development Setup

#### Manual Install

The "Easy Way" is to use the Frappe Bench install script (which installs dependencies such as MariaDB).  See [bench](https://github.com/frappe/bench) for more details.

The install script creates new passwords for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user. The passwords are saved to `~/frappe_passwords.txt`.

#### Local Development

To set up the repository locally:

1.  Set up bench (see [Installation Steps](https://frappeframework.com/docs/user/en/installation)
2.  Start the server:
    ```bash
    bench start
    ```
3.  Open a new terminal window and run:
    ```bash
    bench new-site erpnext.localhost
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
4.  Access the app: `http://erpnext.localhost:8000/app` in your browser.

## Resources and Community

*   [Frappe School](https://school.frappe.io) - Learn ERPNext and Frappe Framework.
*   [Official Documentation](https://docs.erpnext.com/) - Extensive documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Trademark Policy

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
```
Key improvements and explanations:

*   **SEO Optimization:**  The title is changed to include more relevant keywords ("Open-Source ERP," "Growing Businesses") and the intro paragraph clearly states what ERPNext *is*.  More descriptive headers are used.
*   **One-Sentence Hook:** The introductory paragraph now opens with a compelling sentence to grab the user's attention.
*   **Clear Headings:**  Improved heading structure (using `##` for main sections) to improve readability and organization for search engines.
*   **Bulleted Key Features:**  Uses bullet points to highlight the main features, making them easy to scan.
*   **Actionable Language:** Uses verbs like "Empower," "Manage," "Streamline" to encourage engagement.
*   **Links to Key Resources:** Added a link to the GitHub repository.
*   **Clearer Instructions:** Minor improvements to the Docker installation instructions and local development setup.
*   **Concise Descriptions:** Improved descriptions for each section and key feature.
*   **Focus on Benefits:** Emphasizes the benefits of open-source and ERPNext's capabilities.
*   **Community & Support Emphasis:** Highlights learning resources and community engagement opportunities.
*   **Clearer Call to Action:** Provides explicit instructions to help users start.