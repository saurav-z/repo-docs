<div align="center">
    <a href="https://frappe.io/erpnext">
	    <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
</div>

## ERPNext: The Open-Source ERP That Empowers Your Business

ERPNext is a powerful, intuitive, and **open-source ERP system** designed to streamline your business operations.  Manage everything from accounting and manufacturing to customer relationships with ease. ([View on GitHub](https://github.com/frappe/erpnext))

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

### Key Features

*   **Accounting:** Manage your finances with comprehensive tools, from transaction recording to financial reporting.
*   **Order Management:** Track inventory, manage sales orders, customers, suppliers, and fulfillment.
*   **Manufacturing:** Simplify production cycles, track material consumption, and improve capacity planning.
*   **Asset Management:** Track assets throughout their lifecycle, from purchase to disposal.
*   **Projects:** Deliver projects on time, within budget, and profitably with project tracking, timesheets, and issue management.

<details open>
    <summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Under the Hood

*   **Frappe Framework:** A full-stack web application framework written in Python and Javascript that provides a robust foundation for building web applications.
*   **Frappe UI:** A Vue-based UI library to provide a modern user interface.

### Production Setup

#### Managed Hosting

Experience the convenience of [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. It handles installation, upgrades, and maintenance, allowing you to focus on your business.

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

**Prerequisites:** docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

1.  Clone the repository:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
```

2.  Run the Docker Compose command:

```bash
docker compose -f pwd.yml up -d
```

After a few minutes, access your site on `localhost:8080`. Use the following default credentials:

*   **Username:** Administrator
*   **Password:** admin

For ARM-based Docker setup, see [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

### Development Setup

#### Manual Install

Follow the steps below to set up the repository locally:

1.  Install dependencies and start the server:

```bash
bench start
```

2.  In a separate terminal, create a new site:

```bash
bench new-site erpnext.localhost
```

3.  Get and install the ERPNext app:

```bash
bench get-app https://github.com/frappe/erpnext
bench --site erpnext.localhost install-app erpnext
```

4.  Open `http://erpnext.localhost:8000/app` in your browser to view the running app.

### Learning and Community

*   [Frappe School](https://school.frappe.io): Learn Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.erpnext.com/): Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me): Get instant help from other users.

### Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

### Logo and Trademark Policy

Review our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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

*   **SEO-optimized title:** Uses strong keywords like "Open-Source ERP" and "ERP System".
*   **One-sentence hook:**  Immediately grabs the reader's attention and introduces the core value proposition.
*   **Clear Headings:** Organizes the information logically.
*   **Bulleted Key Features:** Makes the key benefits easy to scan and understand.
*   **Concise Language:**  Avoids unnecessary jargon.
*   **Actionable Instructions:** Docker setup and development setup steps are clear and direct.
*   **Community Links:** Includes links to important community resources.
*   **Call to Action:** Encourages exploration with links to a demo and website.
*   **GitHub Link:** A prominent link back to the original repository for easy access.
*   **Maintain Original Content:** Preserves the core information from the original README while enhancing its clarity and impact.
*   **Emphasis on Open-Source:**  Highlights the open-source nature of the software, which is a key selling point.