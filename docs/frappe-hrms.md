<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open-Source HR and Payroll Software</h2>
	<p align="center">
		<p><b>Transform your HR processes with Frappe HR, a modern, open-source solution designed for efficiency and employee satisfaction.</b></p>
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
    - <a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## About Frappe HR

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline and optimize your HR operations.  Built on the robust Frappe Framework, it offers a modern, user-friendly interface and a wide array of features to manage the entire employee lifecycle.

## Key Features of Frappe HR

Frappe HR simplifies and automates key HR functions, boosting efficiency and employee satisfaction.

*   **Employee Lifecycle Management:** Easily onboard, manage promotions/transfers, and conduct exit interviews, providing a streamlined experience for employees.
*   **Leave and Attendance Tracking:**  Configure flexible leave policies, integrate regional holidays, utilize geolocation for check-in/check-out, and generate detailed attendance reports.
*   **Expense Claims and Advances:** Manage employee advances, streamline expense claims with multi-level approval workflows, and integrate seamlessly with ERPNext accounting.
*   **Performance Management:**  Set and track goals, align them with Key Result Areas (KRAs), empower employees with self-evaluation tools, and simplify performance appraisal cycles.
*   **Payroll and Taxation:** Create salary structures, configure tax slabs, run payroll, handle additional payments, and generate clear salary slips.
*   **Frappe HR Mobile App:**  Access key HR functions on the go, including leave applications/approvals, check-in/check-out, and employee profile access.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Technologies Under the Hood

Frappe HR is built upon powerful open-source technologies:

*   **Frappe Framework:** A full-stack web application framework (Python and Javascript) providing the foundation for web app development, with database handling, user auth and REST APIs.
*   **Frappe UI:** A Vue-based UI library offering a modern and responsive user interface for building single-page apps.

## Production Setup

### Managed Hosting with Frappe Cloud

Simplify deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.  It handles installation, upgrades, monitoring, and support.

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

1.  **Prerequisites:**  Install Docker, docker-compose, and Git.
2.  **Commands:**

    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```

3.  **Access:**  After setup, access Frappe HR at `http://localhost:8000`.
4.  **Login Credentials:**

    *   Username: `Administrator`
    *   Password: `admin`

### Local Setup

1.  **Bench Setup:** Follow [Frappe Framework Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:

    ```bash
    $ bench start
    ```

2.  **New Site Setup:**

    ```bash
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```

3.  **Access the Site:** Open `http://hrms.local:8080` in your browser.

## Learning and Community Resources

*   [Frappe School](https://frappe.school) - Learn the Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get real-time help from users.

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