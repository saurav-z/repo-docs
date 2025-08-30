<!-- Improved & Summarized README for ERPNext -->

<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Growing Businesses</h2>
</div>

<p align="center">
    <b>Empower your business with ERPNext, the powerful and intuitive open-source ERP (Enterprise Resource Planning) system.</b>
</p>

<div align="center">
    <a href="https://frappe.school"><img src="https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square" alt="Learn ERPNext on Frappe School"></a>
    <br><br>
    <a href="https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml"><img src="https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule" alt="CI Status"></a>
    <a href="https://hub.docker.com/r/frappe/erpnext-worker"><img src="https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg" alt="Docker Pulls"></a>
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
</div>

## What is ERPNext?

ERPNext is a 100% open-source ERP system designed to streamline and manage all aspects of your business operations. From accounting to manufacturing, ERPNext offers a comprehensive suite of modules to help you grow.  This project is actively maintained on [GitHub](https://github.com/frappe/erpnext).

### Key Features:

*   **Accounting:**  Manage your finances with tools for transactions, reporting, and analysis.
*   **Order Management:** Track inventory, handle sales orders, manage suppliers, and streamline fulfillment.
*   **Manufacturing:** Simplify production cycles, track material usage, and manage subcontracting.
*   **Asset Management:**  Track assets from purchase to disposal.
*   **Projects:** Manage internal and external projects, track tasks, and monitor profitability.

<details open>
  <summary>More Features</summary>
    <img src="https://erpnext.com/files/v16_bom.png" alt="BOM"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary"/>
    <img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card"/>
    <img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks"/>
</details>

### Built With:

*   **Frappe Framework:** A robust, full-stack web application framework (Python and Javascript) providing the foundation for ERPNext. ([Frappe Framework GitHub](https://github.com/frappe/frappe))
*   **Frappe UI:** A modern, Vue-based UI library for a user-friendly experience. ([Frappe UI GitHub](https://github.com/frappe/frappe-ui))

## Getting Started

### Production Setup

Choose the hosting option that best suits your needs:

*   **Managed Hosting (Recommended):**  [Frappe Cloud](https://frappecloud.com) offers a hassle-free way to host your ERPNext instance, handling installation, upgrades, and maintenance.

    <div>
        <a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
            <picture>
                <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
                <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
            </picture>
        </a>
    </div>

*   **Self-Hosted:** Deploy ERPNext on your own infrastructure.

    #### Docker

    1.  **Prerequisites:** Docker, docker-compose, git.
    2.  **Run:**

        ```bash
        git clone https://github.com/frappe/frappe_docker
        cd frappe_docker
        docker compose -f pwd.yml up -d
        ```

    3.  **Access:** Your site will be available at `localhost:8080`.
    4.  **Login:** Use the default credentials: Username: `Administrator`, Password: `admin`.
    5. **ARM Setup:** See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions)

### Development Setup

#### Manual Install
1.  **Prerequisites:** Install bench. See [Installation Steps](https://frappeframework.com/docs/user/en/installation)
2.  **Start Server:**
    ```bash
    bench start
    ```
3.  **Create Site:**
    ```bash
    bench new-site erpnext.localhost
    ```
4.  **Get and Install App:**
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
5.  **Access:** Open `http://erpnext.localhost:8000/app` in your browser.

## Resources and Community

*   **Frappe School:**  Learn ERPNext and the Frappe Framework. ([Frappe School](https://school.frappe.io))
*   **Documentation:** Comprehensive ERPNext documentation. ([Official documentation](https://docs.erpnext.com/))
*   **Discussion Forum:** Engage with the ERPNext community. ([Discussion Forum](https://discuss.erpnext.com/))
*   **Telegram Group:** Get instant help from other users. ([Telegram Group](https://erpnext_public.t.me))

## Contributing

We welcome contributions!  See our guidelines:

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