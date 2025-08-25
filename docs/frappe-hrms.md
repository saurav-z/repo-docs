<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HR & Payroll Software</h2>
	<p align="center">
		<p><b>Drive excellence in your company with Frappe HR, a modern, open-source HRMS solution packed with powerful features.</b></p>
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

## Frappe HR: Your Complete HRMS Solution

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline your HR processes and empower your team.  This robust platform offers everything you need to manage your employees from onboarding to payroll, including 13+ modules.  Build your HR system your way!

## Key Features of Frappe HR

*   **Employee Lifecycle Management:**  Manage the entire employee journey, from onboarding and promotions to transfers and exit interviews, all in one place.
*   **Leave and Attendance Tracking:** Configure leave policies, automate holiday schedules, track check-ins/check-outs with geolocation, and monitor leave balances with detailed reporting.
*   **Expense Claims and Advances:** Simplify expense reporting with customizable approval workflows and seamless integration with ERPNext accounting.
*   **Performance Management:**  Set and track goals, align them with key result areas (KRAs), and facilitate appraisal cycles.
*   **Payroll and Taxation:**  Create flexible salary structures, configure tax slabs, process payroll, handle additional payments, and generate clear salary slips.
*   **Frappe HR Mobile App:**  Stay connected with the mobile app: Apply for and approve leaves, check in and out, and access employee profiles on the go.

<details open>
<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

## Under the Hood:  Technology Stack

*   **Frappe Framework:** A powerful Python and JavaScript full-stack web application framework, providing a robust foundation.
*   **Frappe UI:** A modern, Vue-based UI library for a user-friendly experience.

## Production Setup and Deployment

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting your Frappe applications.  It handles installation, updates, monitoring, and maintenance, allowing you to focus on your HR needs.

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

1.  Ensure you have Docker and docker-compose installed.
2.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/hrms
    ```
3.  Navigate to the docker directory:
    ```bash
    cd hrms/docker
    ```
4.  Run the setup:
    ```bash
    docker-compose up
    ```
5.  Access the application at `http://localhost:8000`.
6.  Log in with:
    -   Username: `Administrator`
    -   Password: `admin`

### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server and keep it running
	```sh
	$ bench start
	```
2.  In a separate terminal window, run the following commands
	```sh
	$ bench new-site hrms.local
	$ bench get-app erpnext
	$ bench get-app hrms
	$ bench --site hrms.local install-app hrms
	$ bench --site hrms.local add-to-hosts
	```
3.  You can access the site at `http://hrms.local:8080`

## Resources and Community

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get instant help from users.

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