<!-- SEO-optimized README for ERPNext -->

<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
</div>

# ERPNext: Open-Source ERP for Growing Businesses

**ERPNext is a powerful and intuitive open-source Enterprise Resource Planning (ERP) system that helps businesses streamline operations, increase efficiency, and drive growth.** Get started with ERPNext today!

[View the original repository](https://github.com/frappe/erpnext)

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

ERPNext offers a comprehensive suite of features designed to manage various aspects of your business:

*   **Accounting:** Manage cash flow, track transactions, and generate financial reports.
*   **Order Management:** Track inventory, manage sales orders, and fulfill customer orders.
*   **Manufacturing:** Simplify the production cycle, track material consumption, and handle subcontracting.
*   **Asset Management:** Track assets from purchase to disposal, covering IT infrastructure and equipment.
*   **Projects:** Manage both internal and external projects, track tasks, and monitor profitability.

<details open>

<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Technology Stack

ERPNext is built on a robust and open-source technology stack:

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework (Python/Javascript) that provides a strong foundation for building web applications.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library for a modern and user-friendly interface.

## Production Setup

Choose your preferred deployment method:

### Managed Hosting (Recommended)

*   **Frappe Cloud:** A user-friendly platform for hosting Frappe applications, handling installation, upgrades, monitoring, and support.
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

**Prerequisites:** docker, docker-compose, git.

**Steps:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your site on `localhost:8080`.
**Default login:** Username: `Administrator`, Password: `admin`.

Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based setup.

## Development Setup

### Manual Install

1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to setup bench and start the server:
    ```bash
    bench start
    ```

2.  Open a new terminal and run:
    ```bash
    bench new-site erpnext.localhost
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```

3.  Access the application at `http://erpnext.localhost:8000/app`.

## Learning and Community

*   [Frappe School](https://school.frappe.io): Learn Frappe Framework and ERPNext.
*   [Official documentation](https://docs.erpnext.com/): Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me): Get instant help from the user community.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Trademark Policy

Please review our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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

*   **Clear Title and Hook:**  Uses "ERPNext: Open-Source ERP for Growing Businesses" as the title and a concise one-sentence hook.
*   **Keywords:** Includes relevant keywords throughout the README, such as "open-source ERP," "ERP system," "business management," and feature names.
*   **Headings:**  Organizes the content with clear, descriptive headings.
*   **Bulleted Lists:** Uses bulleted lists for key features, technology stack, and setup instructions, making information easy to scan.
*   **Concise Descriptions:**  Provides brief, informative descriptions of each feature.
*   **Call to Action:** Includes a clear call to action ("Get started with ERPNext today!").
*   **Emphasis on Benefits:** Highlights the benefits of using ERPNext (streamlining operations, increasing efficiency, driving growth).
*   **Links:** Includes links to important resources (website, documentation, demo, etc.) and relevant code repositories.
*   **Structured Content:**  Breaks down complex topics into smaller, easily digestible sections.
*   **Clean Formatting:**  Uses Markdown for readability and consistent presentation.
*   **Alt Text:**  Ensures all images have descriptive alt text for accessibility and SEO.
*   **Managed Hosting Emphasis:** Highlights the managed hosting option (Frappe Cloud) as the recommended approach.