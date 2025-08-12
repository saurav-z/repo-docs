<!-- ERPNext: Open-Source ERP for Business -->

# ERPNext: Open-Source ERP Software for Modern Businesses

**Transform your business operations with ERPNext, the powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system.** ([View the Original Repo](https://github.com/frappe/erpnext))

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

<div align="center">
    <a href="https://frappe.io/erpnext">
	    <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
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

## Key Features of ERPNext

ERPNext provides a comprehensive suite of features to manage and streamline your business operations:

*   **Accounting:** Manage your finances with tools for transactions, financial reporting, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales orders, customers, suppliers, shipments, and order fulfillment.
*   **Manufacturing:** Simplify production cycles, manage material consumption, and handle subcontracting.
*   **Asset Management:**  Track assets from purchase to disposal across your organization.
*   **Projects:** Manage both internal and external projects, tracking tasks, timesheets, and issues for profitability.

<details open>

<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Under the Hood

ERPNext is built upon a powerful foundation:

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework written in Python and Javascript, providing the core infrastructure.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library for a modern and intuitive user interface.

## Getting Started

### Production Setup

Choose the deployment method that best suits your needs:

*   **Managed Hosting (Recommended):**  [Frappe Cloud](https://frappecloud.com) offers a hassle-free way to host and manage your ERPNext instance.  It takes care of installation, setup, upgrades, monitoring, maintenance and support of your Frappe deployments.
    <div>
    	<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
    		<picture>
    			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
    			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
    		</picture>
    	</a>
    </div>
*   **Self-Hosted (Docker):** Deploy ERPNext using Docker for greater control and flexibility.

    **Prerequisites:** Docker, Docker Compose, Git.

    **Steps:**

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    docker compose -f pwd.yml up -d
    ```

    Your site will be accessible on localhost port 8080.

    **Default Login:**
        *   Username: Administrator
        *   Password: admin

    See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

### Development Setup

*   **Manual Install**
    1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up Bench.
        *   ```bash
            bench start
            ```
    2.  In a separate terminal window, run the following commands:
        *   ```bash
            # Create a new site
            bench new-site erpnext.localhost
            ```
        *   ```bash
            # Get the ERPNext app
            bench get-app https://github.com/frappe/erpnext

            # Install the app
            bench --site erpnext.localhost install-app erpnext
            ```
    3.  Open the URL `http://erpnext.localhost:8000/app` in your browser

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext through courses.
*   [Official Documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from the user community.

## Contributing

Help improve ERPNext!

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

*   **Clear Title:**  The title is optimized for search (ERPNext: Open-Source ERP Software).
*   **One-Sentence Hook:**  Provides a concise description to attract users.
*   **Targeted Keywords:**  Uses relevant keywords like "open-source ERP," "ERP software," and feature names.
*   **Structured Headings:**  Uses headings (H1, H2, H3) for readability and SEO.
*   **Bulleted Lists:**  Easy to scan and highlight key features.
*   **Internal Links:**  Links to internal documentation/resources.
*   **External Links:** Includes links to the demo, website and documentation.
*   **Concise Language:**  Streamlined descriptions for better understanding.
*   **Call to action:**  Encourages action (try ERPNext).
*   **Emphasis on Open Source:** Highlights the open-source nature for attracting potential users.
*   **Removed redundant "Motivation" section** - the key benefits were already covered in the key features and the intro.
*   **Refactored "Production Setup" and "Development Setup" sections**:  Improved clarity and added a Docker section with the key commands.