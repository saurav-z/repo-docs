<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HR and Payroll Software</h2>
	<p align="center">
		Manage your entire HR and payroll operations with ease using Frappe HR, a modern, open-source solution.
	</p>

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

</div>

<div align="center">
	<img src=".github/hrms-hero.png"/>
</div>

<div align="center">
	<a href="https://frappe.io/hr">Website</a>
	-
	<a href="https://docs.frappe.io/hr/introduction">Documentation</a>
</div>

## Frappe HR: Your Complete Open-Source HRMS Solution

Frappe HR is a comprehensive, open-source Human Resource Management System (HRMS) designed to streamline and automate your HR and payroll processes.  With over 13 modules, it empowers businesses to efficiently manage their workforce, from employee onboarding to payroll and everything in between.  **[Explore Frappe HR on GitHub](https://github.com/frappe/hrms).**

## Key Features

*   **Employee Lifecycle Management:**  Manage employees throughout their entire journey, including onboarding, promotions, transfers, and performance feedback, creating a smooth employee experience.
*   **Leave and Attendance Tracking:** Configure flexible leave policies, automatically apply regional holidays, utilize geolocation check-in/check-out, and generate comprehensive attendance reports.
*   **Expense Claims and Advances:**  Efficiently manage employee advances, expense claims with multi-level approval workflows, and integrate seamlessly with ERPNext accounting.
*   **Performance Management:**  Set and track employee goals, align them with key result areas (KRAs), facilitate self-evaluations, and simplify appraisal cycles.
*   **Payroll & Taxation:**  Create flexible salary structures, configure income tax slabs, run standard payroll runs, handle additional payments, and generate detailed salary slips.
*   **Frappe HR Mobile App:**  Empower employees to apply for and approve leaves, check-in/check-out, and access their profile directly from the mobile app.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Under the Hood

*   **Frappe Framework:** A robust, full-stack web application framework built with Python and JavaScript, providing the foundation for Frappe HR with features like database abstraction, user authentication, and a REST API.
*   **Frappe UI:** A modern and user-friendly Vue.js-based UI library that provides the visual components for building single-page applications on top of the Frappe Framework.

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. Frappe Cloud handles installation, upgrades, monitoring, and support, offering a comprehensive developer platform to manage your Frappe deployments.

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

Requires Docker, docker-compose, and git. Follow these steps:

1.  Clone the repository: `git clone https://github.com/frappe/hrms`
2.  Navigate to the Docker directory: `cd hrms/docker`
3.  Run Docker Compose: `docker-compose up`

Access the application at `http://localhost:8000` using the credentials:

*   Username: `Administrator`
*   Password: `admin`

### Local

1.  Set up bench using the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server: `$ bench start`
2.  Open a new terminal and run the following commands:
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
2.  [Documentation](https://docs.frappe.io/hr) - Comprehensive Frappe HR documentation.
3.  [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://t.me/frappehr) - Get instant support.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

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