<div align="center">
    <a href="https://frappe.io/erpnext">
	    <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
</div>

# ERPNext: Open-Source ERP for Your Business

**Streamline your operations and boost efficiency with ERPNext, a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system.** ([View on GitHub](https://github.com/frappe/erpnext))

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

ERPNext is a comprehensive ERP solution designed to manage all aspects of your business.

*   **Accounting:** Manage cash flow, track transactions, and generate financial reports.
*   **Order Management:** Track inventory, manage sales orders, and handle order fulfillment.
*   **Manufacturing:** Simplify the production cycle, track material consumption, and manage subcontracting.
*   **Asset Management:** Track assets from purchase to disposal, across your entire organization.
*   **Projects:** Manage both internal and external projects, track tasks, and analyze project profitability.

<details open>
    <summary>More Features</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Technical Underpinnings

*   **Frappe Framework:** A full-stack web application framework (Python/Javascript) providing the foundation for ERPNext.  ([Frappe Framework](https://github.com/frappe/frappe))
*   **Frappe UI:**  A Vue-based UI library providing a modern and intuitive user interface.  ([Frappe UI](https://github.com/frappe/frappe-ui))

## Get Started with ERPNext

### Managed Hosting

Simplify your ERPNext deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hassle-free hosting, maintenance, and support.

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

1.  **Prerequisites:** Docker, docker-compose, git.
2.  **Clone and Run:**

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    docker compose -f pwd.yml up -d
    ```

3.  **Access:**  Your ERPNext site will be accessible on `localhost:8080`.
4.  **Credentials:**  Use the default login credentials:  Username: `Administrator`, Password: `admin`.

   For ARM-based Docker setup, refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions)

#### Development Setup

1.  **Manual Install**
   See [Frappe Bench documentation](https://github.com/frappe/bench) for manual install.
   The script creates new passwords for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user.
2.  **Local Setup**

    1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up bench.

    2.  Start the server `bench start`
    3.  In a separate terminal, run:

        ```bash
        bench new-site erpnext.localhost
        bench get-app https://github.com/frappe/erpnext
        bench --site erpnext.localhost install-app erpnext
        ```

    4.  Open `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community

*   [Frappe School](https://school.frappe.io): Learn ERPNext from the experts.
*   [Official Documentation](https://docs.erpnext.com/):  Comprehensive documentation.
*   [Discussion Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me): Get instant help from users.

## Contributing

We welcome contributions to ERPNext!  Please review these guidelines:

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
```
Key improvements and SEO considerations:

*   **Clear Heading Structure:** Uses `<h1>`, `<h2>`, and `<h3>` for better organization and SEO.
*   **SEO-Optimized Title:**  "ERPNext: Open-Source ERP for Your Business" is a strong title.
*   **One-Sentence Hook:**  Provides a concise and compelling introduction to the project.
*   **Keyword Integration:** Naturally incorporates relevant keywords like "ERP," "open-source," "business," and key features.
*   **Bullet Points:**  Uses bulleted lists to highlight key features, making them easily scannable.
*   **Clear Sections:** Separates content into logical sections (Features, Setup, Learning, Contributing).
*   **Links to Resources:**  Includes links to documentation, the website, the demo, and the original GitHub repo.
*   **Call to Action:** Encourages users to try the demo, visit the website, and contribute.
*   **Readability:** Improves readability with better formatting and spacing.
*   **Focus on Value Proposition:** Emphasizes the benefits of ERPNext (streamlining operations, boosting efficiency, cost-effective).
*   **Searchable Keywords:** Uses common search terms to improve search engine visibility.
*   **Contextual Images:** Keeps the images from the original readme for visual appeal.