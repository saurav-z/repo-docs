<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HR & Payroll Software</h2>
	<p align="center">
		<p>Manage your entire employee lifecycle with Frappe HR – a powerful, open-source HRMS solution.</p>
	</p>

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

<a href="https://trendshift.io/repositories/10972" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10972" alt="frappe%2Fhrms | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<div align="center">
	<img src=".github/hrms-hero.png"/>
</div>

<div align="center">
	<a href="https://frappe.io/hr">Website</a>
	-
	<a href="https://docs.frappe.io/hr/introduction">Documentation</a>
	-
	<a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## Frappe HR: Your Complete Open-Source HRMS Solution

Frappe HR empowers businesses with a comprehensive, modern, and easy-to-use HR and Payroll software solution. Built on the Frappe framework, it offers a complete HRMS solution with over 13 different modules to streamline your HR processes. **[Learn more on GitHub](https://github.com/frappe/hrms)**

## Key Features

*   **Employee Lifecycle Management:** From onboarding to offboarding, Frappe HR simplifies the employee journey, managing promotions, transfers, and feedback.
*   **Leave and Attendance Tracking:** Configure leave policies, manage attendance with geolocation, and track leave balances with comprehensive reporting.
*   **Expense Claim & Advance Management:** Manage employee advances, claim expenses, and automate approval workflows, seamlessly integrating with ERPNext accounting.
*   **Performance Management:** Track goals, align them with key result areas (KRAs), and streamline the appraisal cycle.
*   **Payroll & Taxation:** Create salary structures, configure tax slabs, process payroll, handle additional payments, and generate detailed salary slips.
*   **Mobile App:** Access HR functions on the go with the Frappe HR mobile app, allowing employees to apply for leaves, check-in/out, and view profiles.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

## Under the Hood

*   **Frappe Framework:** A robust, full-stack web application framework (Python & Javascript) providing the foundation.
*   **Frappe UI:** A modern, Vue.js-based UI library for a seamless user experience.

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hassle-free Frappe application hosting.

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

1.  Install Docker, Docker Compose, and Git.
2.  Clone the repository: `git clone https://github.com/frappe/hrms`
3.  Navigate to the Docker directory: `cd hrms/docker`
4.  Run: `docker-compose up`
5.  Access the HR application at `http://localhost:8000` with the following credentials:
    *   Username: `Administrator`
    *   Password: `admin`

### Local Setup

1.  Set up bench following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server with: `$ bench start`
2.  In a separate terminal window:
    ```sh
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
2.  [Documentation](https://docs.frappe.io/hr)
3.  [User Forum](https://discuss.erpnext.com/)
4.  [Telegram Group](https://t.me/frappehr)

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

Key improvements and SEO considerations:

*   **Clear Title with Keywords:**  "Frappe HR: Open Source HR & Payroll Software" uses the primary keywords.
*   **Compelling Hook:** A concise sentence that immediately captures the user's attention.
*   **Strategic Keyword Placement:** Keywords like "HRMS," "open source," "payroll," "employee lifecycle" are used naturally throughout.
*   **Bulleted Key Features:**  Easy-to-scan format for quick understanding.
*   **Internal Linking:** Added a "View on GitHub" link for better SEO and navigation.
*   **Stronger Headings:**  Clear and descriptive headings for improved readability and SEO.
*   **Concise Language:**  Avoids unnecessary jargon.
*   **Call to Action:** Encourages users to learn more on GitHub.
*   **Markdown Formatting:**  Uses proper Markdown for optimal rendering and SEO benefits.
*   **Removed Redundancy:** Streamlined text for clarity.