# ERPNext: Open-Source ERP for Growing Businesses

**Revolutionize your business operations with ERPNext, a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system.**  [Explore the original repository](https://github.com/frappe/erpnext)

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

![ERPNext Hero Image](https://github.com/frappe/erpnext/raw/develop/erpnext/public/images/v16/hero_image.png)

*   [Live Demo](https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo)
*   [Website](https://frappe.io/erpnext)
*   [Documentation](https://docs.frappe.io/erpnext/)

## Key Features of ERPNext

ERPNext is designed to be the central hub for all your business needs. Key features include:

*   **Accounting:** Manage your finances with ease, from transactions to comprehensive financial reports.
*   **Order Management:** Streamline your sales process by tracking inventory, managing orders, and fulfilling deliveries.
*   **Manufacturing:** Simplify production cycles, track material usage, and optimize your manufacturing processes.
*   **Asset Management:** Track and manage your organization's assets, from IT infrastructure to equipment.
*   **Projects:** Deliver projects on time, within budget, and increase profitability with project tracking, task management, and timesheets.

<details>
  <summary>More Features</summary>
  <img src="https://erpnext.com/files/v16_bom.png"/>
  <img src="https://erpnext.com/files/v16_stock_summary.png"/>
  <img src="https://erpnext.com/files/v16_job_card.png"/>
  <img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Under the Hood: Technology Stack

ERPNext is built on a robust open-source foundation:

*   **Frappe Framework:** A full-stack web application framework (Python & Javascript) providing the core infrastructure for ERPNext.
*   **Frappe UI:**  A modern, Vue.js-based UI library for a user-friendly experience.

## Getting Started

### Managed Hosting (Recommended)

For the easiest setup, consider [Frappe Cloud](https://frappecloud.com), a managed hosting platform for Frappe applications.

<div>
	<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

### Self-Hosted (Docker)

1.  **Prerequisites:** Docker, Docker Compose, and Git installed.
2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

3.  **Run with Docker Compose:**

    ```bash
    docker compose -f pwd.yml up -d
    ```

4.  **Access ERPNext:** After a few minutes, your ERPNext instance will be accessible at `http://localhost:8080`.
    *   **Login:** Use the default credentials:
        *   Username: `Administrator`
        *   Password: `admin`
    *   Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

### Development Setup

#### Manual Install

1.  **Bench Setup:** Follow the [installation steps](https://frappeframework.com/docs/user/en/installation) to set up bench.
2.  **Start the Server:**
    ```bash
    bench start
    ```
3.  **Create a Site:**
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

*   **Frappe School:**  Learn from the experts through courses on the Frappe Framework and ERPNext ([https://school.frappe.io](https://school.frappe.io)).
*   **Official Documentation:** Comprehensive documentation ([https://docs.erpnext.com/](https://docs.erpnext.com/))
*   **Discussion Forum:** Engage with the ERPNext community ([https://discuss.erpnext.com/](https://discuss.erpnext.com/))
*   **Telegram Group:**  Get instant help from a large user base ([https://erpnext_public.t.me](https://erpnext_public.t.me))

## Contributing

Contribute to the future of ERPNext!

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

Key improvements and SEO considerations:

*   **Clear Headline:**  Uses "ERPNext" and includes keywords like "Open-Source" and "ERP" for better search visibility.
*   **One-Sentence Hook:** Grabs attention immediately and clearly states the value proposition.
*   **Keyword Optimization:**  Repeats relevant keywords naturally throughout the document.
*   **Structured Content:** Uses headings, subheadings, and bullet points for easy readability and SEO ranking.
*   **Concise Language:**  Avoids unnecessary jargon and focuses on core information.
*   **Call to Action:** Encourages users to explore the demo, website, and documentation.
*   **Internal Linking:** Linking to relevant resources within the documentation.
*   **Alt Text for Images:**  Includes alt text for images to improve accessibility and SEO.
*   **Clear Code Blocks:** Uses code blocks with syntax highlighting for better readability.
*   **Focus on Benefits:**  Highlights the advantages of using ERPNext.
*   **Community & Learning:** Expanded the learning and community section
*   **Managed Hosting:** Gave a more prominent introduction for managed hosting.