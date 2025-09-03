# ERPNext: Open-Source ERP for Business Management

**Tired of juggling multiple software solutions? ERPNext is a powerful, open-source ERP system that streamlines your business operations and empowers growth.**  ([View the Original Repo](https://github.com/frappe/erpnext))

<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
</div>

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

ERPNext offers a comprehensive suite of features to manage all aspects of your business:

*   **Accounting:** Streamline financial management, from transactions to insightful reports.
*   **Order Management:** Efficiently track inventory, manage orders, and fulfill customer needs.
*   **Manufacturing:** Simplify production cycles, monitor material consumption, and optimize capacity planning.
*   **Asset Management:** Track and manage assets, from IT infrastructure to equipment, throughout their lifecycle.
*   **Projects:** Manage projects on time and budget, and track tasks, timesheets, and issues.

<details open>

<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Under the Hood: Built on Powerful Technologies

ERPNext is built on the following technologies:

*   **Frappe Framework:** A full-stack web application framework written in Python and JavaScript, providing a robust foundation. ([Frappe Framework](https://github.com/frappe/frappe))
*   **Frappe UI:** A modern, Vue-based UI library for building intuitive user interfaces. ([Frappe UI](https://github.com/frappe/frappe-ui))

## Production Setup

### Managed Hosting with Frappe Cloud

Get up and running quickly with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. It simplifies installation, upgrades, and maintenance.

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

Prerequisites: docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

Run following commands:

```
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a couple of minutes, site should be accessible on your localhost port: 8080. Use below default login credentials to access the site.
- Username: Administrator
- Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The Easy Way: our install script for bench will install all dependencies (e.g. MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

To setup the repository locally follow the steps mentioned below:

1.  Setup bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server
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

## Learning and Community Resources

*   [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from a large user community.

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