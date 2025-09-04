<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HR and Payroll Software</h2>
	<p align="center">
		<p>Manage your human resources effectively with Frappe HR, a modern, open-source solution designed for ease of use.</p>
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

## Table of Contents
*   [About Frappe HR](#frappe-hr)
*   [Key Features](#key-features)
*   [Screenshots](#screenshots)
*   [Under the Hood](#under-the-hood)
*   [Production Setup](#production-setup)
*   [Development Setup](#development-setup)
*   [Learning and Community](#learning-and-community)
*   [Contributing](#contributing)
*   [Logo and Trademark Policy](#logo-and-trademark-policy)

## Frappe HR

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline your HR processes. This modern software solution offers a suite of over 13 modules, from employee management and onboarding to payroll and taxation, empowering you to drive excellence within your company. Built on the robust Frappe Framework, Frappe HR provides a user-friendly interface and the flexibility to adapt to your specific HR needs.  Originally part of ERPNext, Frappe HR is now a standalone product, offering a dedicated and mature HR management experience.

## Key Features

Frappe HR provides a rich set of features to manage your HR needs efficiently:

*   **Employee Lifecycle Management:** Simplify employee onboarding, manage promotions and transfers, and conduct exit interviews to support employees throughout their career.
*   **Leave and Attendance Management:** Configure leave policies, integrate regional holidays, use geolocation for check-in/check-out, and track leave balances and attendance with comprehensive reports.
*   **Expense Claims and Advances:** Manage employee advances and claims with multi-level approval workflows, seamlessly integrated with ERPNext accounting.
*   **Performance Management:** Track and align goals with key result areas (KRAs), enable employee self-evaluations, and streamline appraisal cycles.
*   **Payroll & Taxation:** Create salary structures, configure income tax slabs, run payroll, handle additional payments, and view income breakdowns on salary slips.
*   **Frappe HR Mobile App:** Apply for and approve leaves, and access employee profiles on the go.

<details open>
<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

## Under the Hood

Frappe HR is built on the following technologies:

*   **Frappe Framework:** A powerful full-stack web application framework written in Python and Javascript. It provides a robust foundation for building web applications.
*   **Frappe UI:** A Vue-based UI library, providing a modern user interface.

## Production Setup

### Managed Hosting

Frappe Cloud offers a simple, user-friendly, and sophisticated platform to host Frappe applications. It handles installation, upgrades, monitoring, and maintenance.

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

1.  Ensure you have Docker, docker-compose, and Git set up on your machine. Refer to the [Docker documentation](https://docs.docker.com/).
2.  Run the following commands:

```bash
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```
Access the application at `http://localhost:8000` with the credentials:
- Username: `Administrator`
- Password: `admin`


### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:

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

Explore resources to learn and engage with the Frappe HR community:

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
2.  [Documentation](https://docs.frappe.io/hr) - Comprehensive documentation for Frappe HR.
3.  [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://t.me/frappehr) - Get instant help from the community.

## Contributing

Contribute to Frappe HR development:

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Logo and Trademark Policy

Review our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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