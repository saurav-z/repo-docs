# ERPNext: Open-Source ERP for Business Management

**Streamline your business operations with ERPNext, a powerful, intuitive, and completely open-source Enterprise Resource Planning (ERP) system.**  ([View the original repository](https://github.com/frappe/erpnext))

<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
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

ERPNext offers a comprehensive suite of features to manage all aspects of your business.

*   **Accounting:** Manage your finances, track cash flow, and generate insightful financial reports.
*   **Order Management:**  Efficiently manage inventory, sales orders, customers, suppliers, and order fulfillment.
*   **Manufacturing:** Simplify your production cycle with features for material consumption, capacity planning, and subcontracting.
*   **Asset Management:** Track your organization's assets, from IT infrastructure to equipment, throughout their lifecycle.
*   **Projects:** Manage both internal and external projects on time and within budget, including task tracking and timesheets.

<details open>

<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Technologies Under the Hood

*   **Frappe Framework:**  A robust, full-stack web application framework built with Python and JavaScript, providing the foundation for ERPNext.
*   **Frappe UI:** A modern and flexible Vue-based UI library that enhances the user experience.

## Getting Started

### Production Setup

Choose the deployment option that best suits your needs:

#### Managed Hosting

Simplify your ERPNext deployment with [Frappe Cloud](https://frappecloud.com). It offers hassle-free hosting, automatic updates, and expert support.

<div>
	<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

#### Self-Hosted

Deploy ERPNext on your own infrastructure.

##### Docker

1.  **Prerequisites:** Ensure you have Docker, Docker Compose, and Git installed.
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
3.  **Run Docker Compose:**
    ```bash
    docker compose -f pwd.yml up -d
    ```
4.  **Access ERPNext:**  After a few minutes, your ERPNext instance will be available at `localhost:8080`.
    *   **Login Credentials:**
        *   Username: `Administrator`
        *   Password: `admin`
5.  **ARM Architecture:**  For ARM-based Docker setups, refer to the [Frappe Docker documentation](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Install

Set up ERPNext locally for development.

1.  **Install Bench:** Follow the [Bench installation steps](https://frappeframework.com/docs/user/en/installation).
2.  **Start the Server:**
    ```bash
    bench start
    ```
3.  **Create a New Site:**
    ```bash
    bench new-site erpnext.localhost
    ```
4.  **Get and Install ERPNext:**
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
5.  **Access the App:** Open `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community Resources

*   **Frappe School:**  Learn ERPNext and the Frappe Framework through courses by the maintainers and community. ([Frappe School](https://school.frappe.io))
*   **Official Documentation:** Comprehensive documentation for ERPNext. ([Official documentation](https://docs.erpnext.com/))
*   **Discussion Forum:**  Connect with the ERPNext community. ([Discussion Forum](https://discuss.erpnext.com/))
*   **Telegram Group:** Get instant help from a large community of users. ([Telegram Group](https://erpnext_public.t.me))

## Contributing

Contribute to the ERPNext project:

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Review the [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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