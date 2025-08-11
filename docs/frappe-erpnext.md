<!-- ERPNext: The Open-Source ERP Solution for Your Business -->
<div align="center">
    <a href="https://frappe.io/erpnext">
	    <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
    <h2>ERPNext</h2>
    <p align="center">
        Powerful, Intuitive, and Open-Source ERP
    </p>

    <p>
        <a href="https://frappe.school"><img src="https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square" alt="Learn on Frappe School"></a><br><br>
        <a href="https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml"><img src="https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule" alt="CI"></a>
        <a href="https://hub.docker.com/r/frappe/erpnext-worker"><img src="https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg" alt="Docker Pulls"></a>
    </p>

    <p>
        <a href="https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo">Live Demo</a> |
        <a href="https://frappe.io/erpnext">Website</a> |
        <a href="https://docs.frappe.io/erpnext/">Documentation</a>
    </p>
</div>

<div align="center">
	<img src="./erpnext/public/images/v16/hero_image.png"/>
</div>

## ERPNext: Run Your Business Better with Open-Source ERP

ERPNext is a powerful, open-source Enterprise Resource Planning (ERP) system designed to streamline your business operations. [Discover ERPNext on GitHub](https://github.com/frappe/erpnext).

### Key Features:

*   **Accounting:** Comprehensive tools to manage your finances, from transactions to financial reports.
*   **Order Management:** Efficiently track inventory, manage sales orders, and fulfill customer needs.
*   **Manufacturing:** Simplify production cycles, track material consumption, and optimize capacity planning.
*   **Asset Management:**  Manage your assets throughout their lifecycle.
*   **Project Management:** Deliver internal and external projects efficiently, on time, and within budget.

<details open>
  <summary>More Visuals</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Under the Hood

*   **Frappe Framework:** The robust, full-stack web application framework (Python and Javascript) powering ERPNext.  [Learn more about Frappe Framework](https://github.com/frappe/frappe).
*   **Frappe UI:** Modern and intuitive Vue-based UI library for a superior user experience.  [Explore Frappe UI](https://github.com/frappe/frappe-ui).

## Production Setup

### Managed Hosting: Frappe Cloud

Simplify your ERPNext deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.  It handles installation, updates, monitoring, and support.

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

**Prerequisites:** Docker, docker-compose, git.  Refer to [Docker Documentation](https://docs.docker.com) for setup.

**Installation:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your site on `localhost:8080` (after a few minutes) with:
*   Username: Administrator
*   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setups.

## Development Setup

### Manual Install

The easiest approach is to use our install script, which will automatically install dependencies (e.g., MariaDB). See [frappe/bench](https://github.com/frappe/bench) for details.  Passwords for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user are automatically generated and saved to `~/frappe_passwords.txt`.

### Local Setup

1.  Set up bench. See [Installation Steps](https://frappeframework.com/docs/user/en/installation)
   ```bash
   bench start
   ```

2.  In a separate terminal:
   ```bash
   bench new-site erpnext.localhost
   bench get-app https://github.com/frappe/erpnext
   bench --site erpnext.localhost install-app erpnext
   ```

3.  Access your app at `http://erpnext.localhost:8000/app`.

## Learning and Community

1.  [Frappe School](https://school.frappe.io):  Learn from courses on Frappe Framework and ERPNext.
2.  [Official documentation](https://docs.erpnext.com/): Comprehensive ERPNext documentation.
3.  [Discussion Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me): Get instant help from other users.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

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