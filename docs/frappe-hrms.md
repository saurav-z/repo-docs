<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HRMS for Modern Businesses</h2>
	<p align="center">
		<p>Open Source, modern, and easy-to-use HR and Payroll Software</p>
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
</div>

## Frappe HR: Your Complete Open Source HRMS Solution

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline all aspects of your HR operations.  From employee management to payroll, Frappe HR provides a modern and user-friendly experience, all while giving you complete control.  **Ready to transform your HR processes?**  Learn more about [Frappe HR on GitHub](https://github.com/frappe/hrms).

## Key Features

Frappe HR offers a robust set of features to manage your entire employee lifecycle:

*   **Employee Lifecycle Management:** Onboard new hires, manage promotions and transfers, and conduct exit interviews seamlessly.
*   **Leave and Attendance Tracking:** Configure flexible leave policies, integrate regional holidays, and track employee attendance with geolocation.
*   **Expense Claims and Advances:**  Manage employee advances, claim expenses, and automate approval workflows with integration to accounting.
*   **Performance Management:** Set and track goals, align them with Key Result Areas (KRAs), and simplify the appraisal process.
*   **Payroll & Taxation:** Create salary structures, configure income tax slabs, run payroll, and generate detailed salary slips.
*   **Frappe HR Mobile App:**  Manage your HR tasks on the go with a mobile app to apply and approve leaves, and access employee profiles.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Under the Hood

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework written in Python and Javascript. The framework provides a robust foundation for building web applications, including a database abstraction layer, user authentication, and a REST API.

*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library, to provide a modern user interface. The Frappe UI library provides a variety of components that can be used to build single-page applications on top of the Frappe Framework.

## Getting Started

### Production Setup

For a hassle-free experience, consider [Frappe Cloud](https://frappecloud.com), a fully managed platform for hosting Frappe applications.  Frappe Cloud handles installation, upgrades, monitoring, and support.

<div>
	<a href="https://frappecloud.com/hrms/signup" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

### Development Setup

#### Docker
1.  Ensure Docker, docker-compose, and git are installed.
2.  Clone the repository and navigate to the `docker` directory:
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    ```
3.  Run `docker-compose up` to build and run the application.
4.  Access the HR instance at `http://localhost:8000` with credentials: Username: `Administrator`, Password: `admin`.

#### Local Setup

1.  Install the Frappe Framework by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
    ```bash
    $ bench start
    ```
2.  In a separate terminal window, run the following commands:
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
*   [Documentation](https://docs.frappe.io/hr): Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr): Get instant help from users.

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