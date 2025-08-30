<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HR & Payroll Software</h2>
	<p align="center">
		<p>Drive HR excellence with a modern, open-source solution designed for efficiency and employee satisfaction.</p>
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

## About Frappe HR

**Frappe HR** is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline your HR processes and boost employee satisfaction. This powerful software provides a complete HR solution with over 13 modules covering every aspect of HR, from employee management and onboarding to payroll, taxation, and more. Built on the robust [Frappe Framework](https://github.com/frappe/frappe), it offers a modern, user-friendly experience.

**Looking for a complete HRMS solution? [Explore the Frappe HR GitHub Repository](https://github.com/frappe/hrms)**

## Key Features

*   ✅ **Employee Lifecycle Management:** Onboard, manage, and develop employees throughout their careers with features for promotions, transfers, feedback, and exit interviews.
*   ✅ **Leave and Attendance Tracking:** Simplify leave management with customizable policies, automated holiday calendars, geolocation check-in/out, and comprehensive attendance reports.
*   ✅ **Expense Claims and Advances:** Manage employee advances and expenses with multi-level approval workflows, integrated seamlessly with ERPNext accounting.
*   ✅ **Performance Management:** Track goals, align KRAs (Key Result Areas), enable employee self-evaluations, and simplify appraisal cycles.
*   ✅ **Payroll & Taxation:** Create flexible salary structures, configure income tax slabs, run payroll efficiently, manage additional payments, and generate detailed salary slips.
*   ✅ **Mobile Access:** Manage HR tasks on the go with the Frappe HR Mobile App, allowing employees to apply for leaves, check-in/out, and access their profiles.

<details open>
<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

## Technology Stack

*   **Frappe Framework:** The full-stack web application framework, written in Python and Javascript, providing the robust foundation for Frappe HR.
*   **Frappe UI:** A modern, Vue.js-based UI library for a sleek and user-friendly interface.

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hassle-free hosting of Frappe applications. It handles installation, upgrades, monitoring, and support.

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

1.  Ensure you have Docker, docker-compose, and Git installed. Refer to the [Docker documentation](https://docs.docker.com/).
2.  Run the following commands:

    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```

3.  Access the application at `http://localhost:8000` with the following credentials:
    *   Username: `Administrator`
    *   Password: `admin`

### Local

1.  Set up bench following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal:

    ```bash
    bench new-site hrms.local
    bench get-app erpnext
    bench get-app hrms
    bench --site hrms.local install-app hrms
    bench --site hrms.local add-to-hosts
    ```

3.  Access the site at `http://hrms.local:8080`

## Community and Resources

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the community.
*   [Telegram Group](https://t.me/frappehr) - Get instant help.

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