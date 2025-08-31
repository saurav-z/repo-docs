# ERPNext: Open-Source ERP for Businesses of All Sizes

**Transform your business with ERPNext, a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system.**

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

[<img src="./erpnext/public/images/v16/hero_image.png" alt="ERPNext Dashboard">](https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo)

*   [Live Demo](https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo)
*   [Website](https://frappe.io/erpnext)
*   [Documentation](https://docs.frappe.io/erpnext/)
*   **[View the source code on GitHub](https://github.com/frappe/erpnext)**

## Key Features of ERPNext

ERPNext is a comprehensive ERP solution offering a wide range of features to streamline your business operations.

*   **Accounting:** Manage your finances with tools for transaction recording, financial reporting, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales orders, suppliers, shipments, and order fulfillment.
*   **Manufacturing:** Simplify production cycles, track material consumption, and manage subcontracting.
*   **Asset Management:** Track assets from purchase to disposal, covering all branches of your organization.
*   **Projects:** Deliver internal and external projects on time and within budget, tracking tasks, timesheets, and issues.

<details open>
<summary>More Features</summary>
    <img src="https://erpnext.com/files/v16_bom.png" alt="BOM">
    <img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary">
    <img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card">
    <img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks">
</details>

## Under the Hood

ERPNext is built on robust open-source technologies:

*   **Frappe Framework:** A full-stack web application framework (Python & Javascript) providing a solid foundation. ([Frappe Framework GitHub](https://github.com/frappe/frappe))
*   **Frappe UI:** A Vue.js-based UI library for a modern user interface. ([Frappe UI GitHub](https://github.com/frappe/frappe-ui))

## Production Setup

Choose the setup that best suits your needs:

### Managed Hosting (Recommended)

[Frappe Cloud](https://frappecloud.com) offers a simple, user-friendly platform for hosting Frappe applications.  It handles installation, upgrades, monitoring, and maintenance.

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

**Prerequisites:** Docker, Docker Compose, Git.

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run Docker Compose:

    ```bash
    docker compose -f pwd.yml up -d
    ```

    Your site will be accessible on `localhost:8080`.  Use the default login credentials:

    *   Username: `Administrator`
    *   Password: `admin`

    Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

**Follow the instructions in the [Frappe Framework Documentation](https://frappeframework.com/docs/user/en/installation) and use `bench` to create sites and install ERPNext.**
### Local

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

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn ERPNext and the Frappe Framework.
*   [Official Documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help.

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
```
Key improvements and SEO considerations:

*   **Clear Hook:**  The one-sentence hook is at the beginning to immediately grab the reader's attention.
*   **Keyword Optimization:**  Includes keywords like "open-source ERP," "ERP system," and key features.
*   **Headings and Structure:** Uses clear headings and subheadings for readability and SEO benefit.
*   **Bulleted Lists:**  Uses bullet points to highlight key features, making the information easy to scan.
*   **Call to Action (CTA):** Encourages users to check out the demo and website.
*   **Internal Linking:**  Links to relevant pages within the documentation and community resources.
*   **External Linking:** Includes relevant links to the underlying frameworks and related resources.
*   **Image Alt Text:**  Added `alt` text to images to improve SEO.
*   **Concise Language:** Streamlined the language for better readability.
*   **Focus on Benefits:**  Highlights the *benefits* of ERPNext, not just the features.
*   **Clear Instructions:** Improves Docker setup instructions for better usability.
*   **Community and Learning:** Includes links to important learning resources for users to find help and learn.