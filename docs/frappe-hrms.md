<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
</div>

# Frappe HR: Open-Source HR and Payroll Software

**Manage your entire employee lifecycle with Frappe HR, a modern, open-source HRMS designed for ease of use and efficiency.** [View the source code on GitHub](https://github.com/frappe/hrms).

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
</div>


## Key Features of Frappe HR

Frappe HR offers a comprehensive suite of modules to streamline your HR processes:

*   **Employee Lifecycle Management:**  Onboarding, promotions, transfers, feedback, and exit interviews.
*   **Leave and Attendance Tracking:**  Configure leave policies, manage holidays, track check-in/check-out with geolocation, and generate attendance reports.
*   **Expense Claims and Advances:**  Handle employee advances, expense claims with multi-level approval workflows, and seamless ERPNext accounting integration.
*   **Performance Management:**  Set and track goals, align with KRAs, conduct employee self-evaluations, and simplify appraisal cycles.
*   **Payroll & Taxation:**  Create salary structures, configure tax slabs, run payroll, and manage off-cycle payments.  View income details on salary slips.
*   **Mobile App:**  Manage HR tasks on the go with the Frappe HR mobile app. Apply for and approve leaves, check-in/check-out, and access employee profiles.

<details open>
<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

## Under the Hood

*   **Frappe Framework:**  A robust, full-stack web application framework built with Python and JavaScript, providing the foundation for Frappe HR.
*   **Frappe UI:**  A Vue-based UI library that delivers a modern and responsive user interface.

## Getting Started

### Production Setup

For easy and hassle-free hosting, consider [Frappe Cloud](https://frappecloud.com), the official platform for Frappe applications. It handles installation, upgrades, and maintenance.

<div>
	<a href="https://frappecloud.com/hrms/signup" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>


### Development Setup

Choose from the following options to set up the development environment for Frappe HR:

#### Docker

1.  Install Docker and Docker Compose.
2.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```
3.  Access the HR instance at `http://localhost:8000` with the following credentials:
    *   Username: `Administrator`
    *   Password: `admin`

#### Local Installation

1.  Set up your bench environment:  Follow the [Frappe Framework installation instructions](https://frappeframework.com/docs/user/en/installation).  Ensure the server is running: `$ bench start`
2.  Open a new terminal and run:
    ```bash
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`

## Learning and Community

*   [Frappe School](https://frappe.school): Learn Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr): Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr): Get instant help from other users.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

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