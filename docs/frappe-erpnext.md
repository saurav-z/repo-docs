<!-- Improved and SEO-Optimized README for ERPNext -->

<div align="center">
    <a href="https://frappe.io/erpnext">
		<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
    <h2>ERPNext - Open Source ERP Software</h2>
    <p align="center">
        <b>Empower your business with ERPNext, a powerful, intuitive, and 100% open-source ERP system.</b>
    </p>

    <p>
        <a href="https://frappe.school"><img src="https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square" alt="Learn on Frappe School"></a>
        <a href="https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml"><img src="https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule" alt="CI"></a>
        <a href="https://hub.docker.com/r/frappe/erpnext-worker"><img src="https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg" alt="docker pulls"></a>
    </p>
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
    <a href="https://github.com/frappe/erpnext"><b>View Source Code on GitHub</b></a>
</div>

## What is ERPNext?

ERPNext is a leading 100% open-source Enterprise Resource Planning (ERP) software designed to streamline and manage all aspects of your business. From accounting and inventory to manufacturing and customer relationship management (CRM), ERPNext provides a comprehensive solution to run your business efficiently.

### Key Features

*   **Accounting:** Manage finances with tools for transactions, reporting, and analysis.
*   **Order Management:** Track inventory, sales orders, and fulfillment, including suppliers and shipments.
*   **Manufacturing:** Simplify production cycles, track materials, and manage capacity planning.
*   **Asset Management:** Track assets, from procurement to disposal, across your organization.
*   **Projects:** Manage projects, track tasks, and monitor profitability.

<details open>
    <summary>More Visuals</summary>
    <img src="https://erpnext.com/files/v16_bom.png" alt="BOM"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary"/>
    <img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card"/>
    <img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks"/>
</details>

### Under the Hood

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework built in Python and JavaScript, providing a robust foundation for ERPNext.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library that offers a modern and user-friendly interface.

## Production Setup

### Managed Hosting

Get started quickly with [Frappe Cloud](https://frappecloud.com), the user-friendly platform to host your Frappe applications. It handles installation, updates, and maintenance for a hassle-free experience.

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

**Prerequisites:** docker, docker-compose, and git.

**Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run docker compose:
    ```bash
    docker compose -f pwd.yml up -d
    ```

Access your site via `localhost:8080`. Use the following default credentials:
*   Username: Administrator
*   Password: admin

For ARM-based docker setup, see [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Install

Install dependencies with the install script for bench. See [Frappe Bench](https://github.com/frappe/bench) for details.
New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

Follow these steps to set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
    ```bash
    bench start
    ```
2.  Open a new terminal and run:
    ```bash
    bench new-site erpnext.localhost
    ```
3.  Get and install the ERPNext app:
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
4.  Access the app in your browser at `http://erpnext.localhost:8000/app`.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.erpnext.com/) - Comprehensive documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from a large community of users.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

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
```

Key improvements and SEO considerations:

*   **Clear Title and Introduction:**  "ERPNext - Open Source ERP Software" immediately clarifies the project. The one-sentence hook captures attention.
*   **Keyword Optimization:**  Includes relevant keywords like "Open Source ERP", "ERP Software", and feature-specific terms (Accounting, Manufacturing, etc.)
*   **Structured Headings:**  Uses clear and descriptive headings (e.g., "What is ERPNext?", "Key Features", "Production Setup") for readability and SEO.
*   **Bulleted Key Features:** Makes it easy for users to quickly grasp the ERPNext's capabilities.
*   **Concise Descriptions:** Explains features and setup instructions in a clear and understandable manner.
*   **Calls to Action:** Encourages users to explore the live demo, website, and documentation.
*   **GitHub Link:**  Crucial to include the link back to the main repo.
*   **Alternative Text for Images:** Added `alt` text to all images to improve accessibility and SEO.
*   **Community and Learning Section:**  Provides links to resources for learning and engaging with the ERPNext community.
*   **Clear Instructions:**  Improved the Docker and Manual Install sections with better formatting and clearer steps.
*   **Trademark Policy Notice:** Includes the trademark policy link.
*   **Concise Summary:** Reduced verbosity while retaining essential information.