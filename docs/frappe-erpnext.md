# ERPNext: Open-Source ERP Software for Business Management

ERPNext is a powerful, intuitive, and open-source Enterprise Resource Planning (ERP) system, offering a comprehensive solution to streamline your business operations.  [Explore the original repository](https://github.com/frappe/erpnext).

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

## Key Features

ERPNext provides a robust and versatile platform to manage various aspects of your business:

*   **Accounting:** Manage your finances with tools for transaction recording, financial reporting, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales orders, customers, suppliers, and order fulfillment.
*   **Manufacturing:** Simplify production cycles, track material consumption, and optimize capacity planning.
*   **Asset Management:** Monitor assets from purchase to disposal, covering infrastructure and equipment.
*   **Projects:** Manage both internal and external projects on time and within budget. Track tasks, timesheets, and issues.

## Why Choose ERPNext?

ERPNext is a fully open-source ERP, offering:

*   **Cost Savings:** No licensing fees.
*   **Flexibility:** Customizable to fit your business needs.
*   **Comprehensive Functionality:**  All-in-one solution covering core business areas.
*   **Community Support:** A thriving community for assistance and collaboration.

## Getting Started

### Demo

*   [Live Demo](https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo)

### Documentation

*   [Official Documentation](https://docs.frappe.io/erpnext/)

### Website

*   [Website](https://frappe.io/erpnext)

## Production Setup

### Managed Hosting (Frappe Cloud)

For ease of use, consider [Frappe Cloud](https://frappecloud.com), which handles installation, upgrades, and maintenance.

<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
        <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
    </picture>
</a>

### Self-Hosted

#### Docker

1.  **Prerequisites:** Docker, Docker Compose, Git.  Refer to [Docker Documentation](https://docs.docker.com) for setup.
2.  **Steps:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your site at `localhost:8080` (default credentials: Administrator/admin).  See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based setup.

## Development Setup

### Manual Install

Follow the instructions for setting up ERPNext locally.

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server.

    ```bash
    bench start
    ```
2.  In a separate terminal window, run the following commands:

    ```bash
    # Create a new site
    bench new-site erpnext.localhost
    ```

3.  Get the ERPNext app and install it.

    ```bash
    # Get the ERPNext app
    bench get-app https://github.com/frappe/erpnext

    # Install the app
    bench --site erpnext.localhost install-app erpnext
    ```

4.  Open the URL `http://erpnext.localhost:8000/app` in your browser, you should see the app running

### Under the Hood

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework written in Python and Javascript.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library, to provide a modern user interface.

## Learning and Community

*   [Frappe School](https://school.frappe.io): Courses on Frappe Framework and ERPNext.
*   [Official documentation](https://docs.erpnext.com/): Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me): Get instant help from the user community.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

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

*   **SEO Optimization:**  Uses headings (H1, H2, etc.) and includes relevant keywords like "ERP," "open-source," and business management in the title and throughout the text.
*   **Concise Hook:** Starts with a clear, one-sentence introduction.
*   **Clear Structure:**  Uses headings and subheadings to organize information, making it easy to scan.
*   **Bulleted Lists:**  Emphasizes key features and benefits with bullet points for readability.
*   **Actionable Information:**  Provides clear instructions for getting started (demo, documentation, setup).
*   **Concise Summaries:** Condensed lengthy sections into easier-to-read summaries.
*   **Removed Redundancy:** Eliminated repeated phrases and unnecessary details.
*   **Enhanced Formatting:** Used bold text to highlight important elements,  and improved code block formatting.
*   **Stronger Call to Action (Implicit):** The structure encourages readers to explore the demo, documentation, and community.
*   **Links:** Maintains all original links.
*   **Community Focus:** Highlights the community resources.