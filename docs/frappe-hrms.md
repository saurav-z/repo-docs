<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HRMS Software</h2>
	<p align="center">
		<p><b>Empower your workforce with Frappe HR, a modern, open-source HR and payroll solution designed for efficiency and growth.</b></p>
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

## Frappe HR: Your Complete HR Management Solution

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) built to streamline HR operations and drive employee success. With over 13 modules, it covers the entire employee lifecycle, from onboarding to payroll and beyond. This project is a part of [Frappe](https://github.com/frappe/hrms).

## Key Features:

*   **Employee Lifecycle Management:** Simplify onboarding, manage promotions and transfers, and document employee feedback with exit interviews.
*   **Leave and Attendance Tracking:** Configure leave policies, manage regional holidays, track check-ins/check-outs with geolocation, and monitor leave balances.
*   **Expense Claims and Advances:** Streamline employee advances, expense claims, and approval workflows with seamless ERPNext accounting integration.
*   **Performance Management:** Track goals, align with key result areas (KRAs), enable self-evaluations, and simplify appraisal cycles.
*   **Payroll & Taxation:** Create flexible salary structures, configure income tax slabs, run payroll, and view detailed income breakdowns on salary slips.
*   **Frappe HR Mobile App:** Manage HR tasks on the go with the mobile app, including leave applications and approvals, and employee profile access.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Under the Hood

*   **Frappe Framework:** The full-stack web application framework that provides a robust foundation. [Learn more](https://github.com/frappe/frappe).
*   **Frappe UI:** A Vue-based UI library for a modern user interface. [Learn more](https://github.com/frappe/frappe-ui).

## Production Setup

### Managed Hosting

Frappe Cloud offers a simple, user-friendly, and sophisticated platform for hosting Frappe applications. It simplifies installation, setup, upgrades, monitoring, maintenance, and support.

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

Requires Docker, docker-compose, and Git. Run the following commands:

```bash
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```

Access at `http://localhost:8000` using:

-   Username: `Administrator`
-   Password: `admin`

### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server.
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
3.  Access the site at `http://hrms.local:8080`

## Learning and Community

1.  [Frappe School](https://frappe.school): Courses to learn Frappe Framework and ERPNext.
2.  [Documentation](https://docs.frappe.io/hr): Comprehensive Frappe HR documentation.
3.  [User Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
4.  [Telegram Group](https://t.me/frappehr): Get instant help from the community.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Logo and Trademark Policy

Please read our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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
Key improvements and SEO optimizations:

*   **Clear Title and Hook:**  "Frappe HR: Open Source HRMS Software" and "Empower your workforce..." provides a clear description and a benefit-driven hook at the top.
*   **Keyword Optimization:** Used terms like "Open Source HRMS," "HR Management System," and HR software throughout the document to improve search visibility.
*   **Bulleted Key Features:**  Uses bullet points for readability and to highlight key benefits.
*   **Concise Language:** Simplified and rephrased sentences for clarity.
*   **Clear Headings:**  Uses headings to logically organize the content.
*   **Internal and External Links:** Added the project link back to GitHub at the beginning and other relevant internal and external links.
*   **Call to Action:**  Implied by highlighting key benefits.
*   **Clean Formatting:** Improved spacing and formatting for better readability.
*   **Focus on Benefits:** Leading with benefits, rather than just features, to attract the target audience.