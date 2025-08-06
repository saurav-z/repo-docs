<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HRMS and Payroll Software</h2>
	<p align="center">
		<p>Empower your workforce with Frappe HR, a complete open-source HR and payroll solution designed for modern businesses.</p>
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
    -  <a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## Frappe HR: Your Comprehensive HRMS Solution

Frappe HR is a powerful, open-source Human Resource Management System (HRMS) designed to streamline your HR processes. With over 13 modules, Frappe HR offers everything you need to manage your employees effectively, from onboarding to payroll. It's built on the robust [Frappe Framework](https://github.com/frappe/frappe) and provides a modern, user-friendly experience.

### Key Features of Frappe HR

*   **Employee Lifecycle Management:** Simplify employee onboarding, manage promotions, transfers, and conduct exit interviews to improve employee experience.
*   **Leave and Attendance Tracking:** Configure leave policies, automate holiday tracking, use geolocation check-in/check-out, and monitor leave balances and attendance with detailed reports.
*   **Expense Claims and Advances:** Manage employee advances, handle expense claims, and implement multi-level approval workflows, seamlessly integrated with ERPNext accounting.
*   **Performance Management:** Set and track goals, align goals with key result areas (KRAs), enable employee self-evaluations, and simplify appraisal cycles.
*   **Payroll & Taxation:** Create flexible salary structures, configure income tax slabs, run payroll, handle additional and off-cycle payments, and provide detailed salary slips.
*   **Frappe HR Mobile App:** Apply for and approve leaves on the go, check-in/check-out, and access employee profiles directly from your mobile device.

<details open>
<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Under the Hood

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework powering Frappe HR.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A modern Vue-based UI library providing a responsive user interface.

## Production Setup

### Managed Hosting

Simplify your Frappe HR deployment with [Frappe Cloud](https://frappecloud.com), a platform for hassle-free hosting of Frappe applications.

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
    cd hrms/docker
    docker-compose up
    ```
3.  Access Frappe HR at `http://localhost:8000` using the credentials:
    *   Username: `Administrator`
    *   Password: `admin`

### Local

1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up bench and start the server.
2.  In a separate terminal, run:
    ```bash
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
4.  [Telegram Group](https://t.me/frappehr) - Get community support.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Logo and Trademark Policy

Review the [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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