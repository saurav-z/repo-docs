<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HRMS for Modern Businesses</h2>
	<p align="center">
		Frappe HR is a complete, open-source HR and Payroll solution designed to streamline your HR operations and boost employee satisfaction.
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

## About Frappe HR

Frappe HR is a powerful, open-source Human Resources Management System (HRMS) designed to simplify and automate your HR processes. Built by the same team that created ERPNext, Frappe HR offers a modern and user-friendly interface with comprehensive features for managing your employees, payroll, and other HR-related tasks. This is the perfect solution for businesses seeking a flexible and cost-effective HR solution.

## Key Features

*   **Employee Lifecycle Management:** Streamline employee onboarding, manage promotions and transfers, and facilitate exit interviews for a smooth employee experience.
*   **Leave and Attendance Tracking:** Configure leave policies, manage attendance with geolocation, and track leave balances with comprehensive reports.
*   **Expense Claims and Advances:** Simplify expense claims and employee advances with multi-level approval workflows and seamless integration with ERPNext accounting.
*   **Performance Management:** Track employee goals, align them with key result areas (KRAs), conduct performance appraisals, and foster employee growth.
*   **Payroll & Taxation:** Easily create salary structures, configure tax slabs, process payroll, manage off-cycle payments, and generate detailed salary slips.
*   **Mobile App:** Access key HR functions on the go with the Frappe HR mobile app, including leave applications and approvals, and employee profile access.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

## Technical Details

Frappe HR is built on top of the robust [Frappe Framework](https://github.com/frappe/frappe), a full-stack web application framework written in Python and JavaScript, and uses the [Frappe UI](https://github.com/frappe/frappe-ui) library for a modern user interface.

## Getting Started

### Production Setup

For an easy and hassle-free deployment, consider [Frappe Cloud](https://frappecloud.com). This platform simplifies installation, upgrades, and maintenance.

<div>
	<a href="https://frappecloud.com/hrms/signup" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

### Development Setup (Docker)

1.  **Prerequisites:** Ensure you have Docker, docker-compose, and Git installed.
2.  **Clone the repository:**

    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```

3.  Access the application at `http://localhost:8000` using the following credentials:
    *   Username: `Administrator`
    *   Password: `admin`

### Development Setup (Local)

1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up bench and start the server:

    ```bash
    $ bench start
    ```

2.  In a separate terminal, run the following commands:

    ```bash
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```

3.  Access the site at `http://hrms.local:8080`

## Learning and Community

*   [Frappe School](https://frappe.school): Learn the Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr): Comprehensive documentation for Frappe HR.
*   [User Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr): Get instant help from the community.

## Contributing

Contribute to the project by following these guidelines:

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