<!-- ERPNext - Open Source ERP -->

<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP Software</h2>
    <p align="center">
        <b>Power your business with ERPNext, the powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system.</b>
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
</div>

---

## About ERPNext

ERPNext is a comprehensive, open-source ERP (Enterprise Resource Planning) system designed to help businesses manage all their core functions in one place. This robust solution offers a cost-effective and efficient way to streamline operations and drive growth.

### Key Features of ERPNext

*   **Accounting:** Manage your finances with powerful tools, including transaction recording, financial reporting, and cash flow analysis.
*   **Order Management:**  Handle sales orders, track inventory, manage customers and suppliers, shipments, and fulfillment.
*   **Manufacturing:** Streamline your production cycle, track material consumption, manage capacity planning, and handle subcontracting.
*   **Asset Management:** Track your organization's assets, from IT infrastructure to equipment, throughout their lifecycle.
*   **Projects:** Manage both internal and external projects, ensuring timely completion, staying within budget, and maximizing profitability.

<details open>
<summary>More Screenshots</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

---

## Technology Stack

ERPNext is built on the Frappe Framework, a powerful open-source web application framework.

*   **[Frappe Framework](https://github.com/frappe/frappe):** A full-stack web application framework (Python/Javascript) providing a robust foundation for building web applications.
*   **[Frappe UI](https://github.com/frappe/frappe-ui):** A Vue-based UI library for modern user interfaces.

---

## Getting Started

### Production Setup

#### Managed Hosting

Experience the ease of use and sophistication of [Frappe Cloud](https://frappecloud.com), a fully-featured platform for hosting and managing your Frappe applications.  Frappe Cloud handles installation, upgrades, monitoring, and maintenance.

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

**Prerequisites:** Docker, Docker Compose, Git. Refer to the [Docker Documentation](https://docs.docker.com) for more details.

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run Docker Compose:
    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, access your site at `localhost:8080`. Use the default login credentials:

*   Username: `Administrator`
*   Password: `admin`

For ARM-based Docker setups, refer to the [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) documentation.

---

### Development Setup

#### Manual Install

To set up the repository locally:

1.  Follow the [Frappe Framework Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up bench and start the server.
    ```bash
    bench start
    ```

2.  In a separate terminal window, run the following commands:
    ```bash
    # Create a new site
    bench new-site erpnext.localhost
    ```

3.  Get the ERPNext app and install it:
    ```bash
    # Get the ERPNext app
    bench get-app https://github.com/frappe/erpnext

    # Install the app
    bench --site erpnext.localhost install-app erpnext
    ```

4.  Open the URL `http://erpnext.localhost:8000/app` in your browser.

---

## Resources and Community

*   [Frappe School](https://school.frappe.io): Learn Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.erpnext.com/): Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me): Get instant help from a large community of users.

---

## Contributing

We welcome contributions!  Please review the following:

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

---

## License & Trademarks

See the [Logo and Trademark Policy](TRADEMARK_POLICY.md).

---

<div align="center" style="padding-top: 0.75rem;">
	<a href="https://frappe.io" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png">
			<img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies" height="28"/>
		</picture>
	</a>
</div>

[Back to the top](#erpnext-open-source-erp-software)
```
Key improvements and SEO optimizations:

*   **Clear Title and Hook:**  The title is improved to include "Open-Source ERP Software". The hook is the opening sentence and summarizes the core value proposition.
*   **Keywords:** Keywords like "Open-Source ERP", "ERP System", and specific feature names are incorporated naturally.
*   **Headings & Structure:**  Clear headings and subheadings are used to improve readability and SEO. This also helps search engines understand the content's structure.
*   **Bulleted Lists:**  Key features are presented in bulleted lists for easy skimming and understanding.
*   **Concise Language:**  The text is streamlined, getting straight to the point.
*   **Internal Links:** Links within the README to other sections, and the "Back to Top" link improve user navigation and SEO.
*   **External Links:** Key external links, including the original repo and demo sites are included.
*   **Alt Text:** Alt text for images are included, which is good practice for accessibility and SEO.
*   **Trademark Policy:** Trademark policy is included.
*   **Mobile-Friendly formatting** using standard markdown.