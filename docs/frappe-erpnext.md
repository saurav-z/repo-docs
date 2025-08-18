<!-- Improved and SEO-Optimized README for ERPNext -->

<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP Software</h2>
    <p align="center">
        <b>Empower your business with ERPNext, the powerful, intuitive, and open-source Enterprise Resource Planning (ERP) system.</b>
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
  -  <a href="https://github.com/frappe/erpnext">View on GitHub</a>
</div>

## ERPNext: Your All-in-One Business Solution

ERPNext is a comprehensive, 100% open-source ERP system designed to streamline and manage every aspect of your business operations. From accounting to manufacturing, ERPNext offers a unified platform for efficiency and growth.

### Key Features

*   ✅ **Accounting:** Simplify financial management with tools for transactions, reporting, and analysis.
*   ✅ **Order Management:** Track inventory, manage sales and purchases, and optimize fulfillment.
*   ✅ **Manufacturing:** Streamline production, track materials, and manage your shop floor.
*   ✅ **Asset Management:** Keep tabs on your assets from purchase to disposal across the organization.
*   ✅ **Projects:** Deliver internal and external projects on time, on budget, and profitably.

<details open>
    <summary>More</summary>
    <img src="https://erpnext.com/files/v16_bom.png"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png"/>
    <img src="https://erpnext.com/files/v16_job_card.png"/>
    <img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Why Choose ERPNext?

*   **Open Source:** Benefit from a transparent and collaborative platform, free from vendor lock-in.
*   **Comprehensive:** Manage all your core business functions in a single system.
*   **Intuitive:** Easy-to-use interface, designed for business users of all technical skill levels.
*   **Scalable:** Grow your business with a flexible ERP solution that adapts to your needs.

### Technology Under the Hood

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework built on Python and Javascript. It provides the foundation for the application.

*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library, providing a modern and intuitive user interface.

## Production Setup

### Managed Hosting

For a hassle-free ERPNext experience, consider [Frappe Cloud](https://frappecloud.com). It provides a fully-managed platform for hosting and managing your ERPNext deployments, taking care of installation, upgrades, monitoring, and support.

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

**Prerequisites:** Docker, docker-compose, git. Refer to [Docker Documentation](https://docs.docker.com) for setup instructions.

**Installation:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Your ERPNext instance will be accessible on `localhost:8080`. Use the following default login credentials:
- Username: Administrator
- Password: admin

For ARM-based Docker setups, please refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Install

Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up bench.
New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local Development

1.  **Setup Bench:** Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  **Create and Install Site:**

    ```bash
    bench new-site erpnext.localhost
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```

3.  **Access ERPNext:** Open `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community

*   [**Frappe School**](https://school.frappe.io): Learn ERPNext and Frappe Framework through courses and tutorials.
*   [**Official Documentation**](https://docs.erpnext.com/): Extensive documentation for ERPNext.
*   [**Discussion Forum**](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [**Telegram Group**](https://erpnext_public.t.me): Get instant help from a large community of users.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## License and Trademark Policy

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

*   **Clear Headline:** The main headline now clearly states the software type.
*   **SEO-Friendly Intro:** Includes target keywords like "Open-Source ERP Software" and a concise, engaging description.
*   **Keyword Optimization:** Keywords are used throughout the README, including in headings and feature descriptions.
*   **Feature Focus:**  Uses bullet points and concise descriptions of key features to make them easy to scan.  Includes a direct call-to-action for each feature (e.g., "simplify financial management").
*   **Structured Sections:** Uses clear headings to organize the information (e.g., "Production Setup," "Development Setup," "Learning and Community").
*   **Call to Actions:** The inclusion of the Live Demo, Website and Documentation links, along with the call to action for each feature all work to improve engagement.
*   **Direct Link Back:** Includes the direct link to the original repository on Github.
*   **Concise Language:** Avoids unnecessary words and phrases to keep the content clear and easy to understand.
*   **Improved Formatting:** Uses bolding, and spacing to enhance readability.
*   **Community and Learning Focus:** Highlights the community resources, which can boost engagement and search rankings.