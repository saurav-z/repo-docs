<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
</div>

## ERPNext: Open-Source ERP for Growing Businesses

**ERPNext** is a powerful, intuitive, and open-source Enterprise Resource Planning (ERP) system designed to help businesses streamline operations and boost efficiency.  [Learn more and contribute on GitHub](https://github.com/frappe/erpnext).

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

### Key Features of ERPNext

*   **Accounting:** Comprehensive tools for managing finances, from transactions to financial reports.
*   **Order Management:**  Track inventory, manage sales and purchase orders, and handle fulfillment.
*   **Manufacturing:** Simplify production cycles, manage materials, and plan capacity.
*   **Asset Management:**  Track assets throughout their lifecycle, from purchase to disposal.
*   **Projects:** Manage both internal and external projects, track tasks, and monitor profitability.

<details open>
<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Technology Under the Hood

*   **Frappe Framework:** A robust, full-stack web application framework (Python and JavaScript).
*   **Frappe UI:** A modern Vue-based UI library for a user-friendly experience.

### Deployment Options

#### Managed Hosting

Simplify your ERPNext deployment with [Frappe Cloud](https://frappecloud.com), a platform that handles installation, upgrades, monitoring, and support.

<div>
	<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

#### Self-Hosted

##### Docker

**Prerequisites:** Docker, docker-compose, git. See [Docker Documentation](https://docs.docker.com) for setup.

**Steps:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your ERPNext site at `http://localhost:8080`.  Use the following default credentials:
*   Username: Administrator
*   Password: admin

For ARM-based Docker setups, see [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

### Development Setup

#### Manual Install

*   Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to setup bench.
*   Create a new site: `bench new-site erpnext.localhost`
*   Get and install the ERPNext app:

```bash
bench get-app https://github.com/frappe/erpnext
bench --site erpnext.localhost install-app erpnext
```
*   Open the URL `http://erpnext.localhost:8000/app` in your browser

### Learning and Community

*   [Frappe School](https://school.frappe.io) -  Learn from courses by maintainers and the community.
*   [Official documentation](https://docs.erpnext.com/)
*   [Discussion Forum](https://discuss.erpnext.com/)
*   [Telegram Group](https://erpnext_public.t.me)

### Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

### Logo and Trademark Policy

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

*   **SEO Optimization:**  The title and headings use keywords like "ERPNext," "Open-Source ERP," and "ERP." This will help with search engine ranking.
*   **Clear Hook:** The one-sentence summary provides a concise introduction to the software.
*   **Structured Content:**  Uses clear headings and subheadings for readability and better SEO.
*   **Bulleted Key Features:** Makes the core functionalities easily scannable.
*   **Concise Explanations:**  Avoids overly verbose descriptions.
*   **Call to Action:**  Includes links to the demo, website, and documentation.
*   **Links Back to Repo:** Includes a link back to the original GitHub repo.
*   **Updated Formatting:** Corrected the markdown formatting to ensure proper rendering.  Removed unnecessary `<div>` tags in favor of markdown's inherent structure.
*   **Docker Instructions Improvement:**  The Docker instructions are streamlined.
*   **Community Resources:** Highlighting links to Frappe School, Documentation, Forum, and Telegram group will make it much easier for the user to find help or connect with other users.
*   **Simplified and Reorganized:** The content is reorganized to flow logically from introduction to setup and contribution guidelines.
*   **Removed Redundancy:**  Eliminated repetition of information where possible.