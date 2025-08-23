<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
</div>

# Frappe HR: Open-Source HRMS Software for Modern Businesses

**Streamline your HR processes and empower your team with Frappe HR, a modern, open-source HR and Payroll solution.**

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

<a href="https://trendshift.io/repositories/10972" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10972" alt="frappe%2Fhrms | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<div align="center">
	<img src=".github/hrms-hero.png"/>
</div>

<div align="center">
	<a href="https://frappe.io/hr">Website</a>
	-
	<a href="https://docs.frappe.io/hr/introduction">Documentation</a>
	-
	<a href="https://github.com/frappe/hrms">**View on GitHub**</a>
</div>

## Key Features of Frappe HR

Frappe HR is a comprehensive HRMS solution packed with features to manage your entire employee lifecycle.

*   **Employee Lifecycle Management:** Simplify onboarding, manage promotions, transfers, and conduct exit interviews to support employees throughout their journey.
*   **Leave and Attendance Tracking:** Configure flexible leave policies, integrate regional holidays, enable geolocation check-in/out, and generate detailed attendance reports.
*   **Expense Claims and Advances:** Manage employee advances, claim expenses with multi-level approval workflows, seamlessly integrated with ERPNext accounting.
*   **Performance Management:** Track goals, align them with key result areas (KRAs), empower employees with self-evaluations, and streamline appraisal cycles.
*   **Payroll & Taxation:** Design salary structures, configure tax slabs, run payroll, accommodate off-cycle payments, and view income breakdowns.
*   **Frappe HR Mobile App:** Access key HR functions on the go, including leave applications, approvals, and employee profile access.

<details open>
<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

## Why Choose Frappe HR?

Frappe HR was built to address the need for truly open-source HR software, and has grown from a set of modules within ERPNext to become a standalone product. It offers a modern, easy-to-use interface built on the robust [Frappe Framework](https://github.com/frappe/frappe) and [Frappe UI](https://github.com/frappe/frappe-ui).

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. Frappe Cloud handles installation, upgrades, monitoring, and support.

<div>
	<a href="https://frappecloud.com/hrms/signup" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

## Development Setup

### Docker

1.  **Prerequisites:** Docker, docker-compose, and git.
2.  **Clone the repository:** `git clone https://github.com/frappe/hrms`
3.  **Navigate to the Docker directory:** `cd hrms/docker`
4.  **Run Docker Compose:** `docker-compose up`
5.  **Access Frappe HR:**  Open `http://localhost:8000` in your browser.

    *   **Login Credentials:**
        *   Username: `Administrator`
        *   Password: `admin`

### Local

1.  **Install Bench:** Follow the [installation steps](https://frappeframework.com/docs/user/en/installation).
2.  **Start Bench:** `bench start`
3.  **Open a new terminal and run:**
    ```sh
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
4.  **Access the site:** `http://hrms.local:8080`

## Learning and Community

*   [Frappe School](https://frappe.school) - Learn about the Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive documentation for Frappe HR.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get real-time help from the user community.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

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
Key improvements and explanations:

*   **SEO-Friendly Title:**  Uses "Frappe HR" and keywords like "Open-Source", "HRMS", and "Software" in the title to improve search visibility.
*   **One-Sentence Hook:** "Streamline your HR processes and empower your team with Frappe HR, a modern, open-source HR and Payroll solution."  This immediately grabs the reader's attention and explains the value proposition.
*   **Clear Headings and Structure:** Uses Markdown headings (H1, H2, etc.) to organize the content logically and improve readability.  This also helps with SEO as search engines use headings to understand the content's structure.
*   **Bulleted Key Features:** Highlights the most important features in an easy-to-scan bulleted list. This is much more user-friendly than a paragraph.
*   **Stronger Keyword Usage:**  Naturally incorporates relevant keywords throughout the document.
*   **Concise Language:**  Streamlined the text to be more direct and to the point.
*   **Call to Action:** Includes links to the website, documentation, and GitHub repo to encourage user engagement.  The "View on GitHub" link is particularly important.
*   **Cleaned up Docker instructions:** The Docker instructions are more concise and direct.
*   **Added "Why Choose Frappe HR?"** This answers the reader's question - why they should choose Frappe HR.
*   **Formatting:**  Uses bolding for emphasis and improved visual appearance.