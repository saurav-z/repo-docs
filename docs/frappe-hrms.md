<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HRMS Solution</h2>
	<p align="center">
		<b>Manage your entire employee lifecycle with Frappe HR, a modern, open-source HR and Payroll software.</b>
	</p>
</div>

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
	-
	<a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## About Frappe HR

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline all aspects of HR operations. From employee onboarding to payroll and performance management, Frappe HR offers a complete suite of tools to empower your HR department.  Built on the Frappe Framework, Frappe HR provides a user-friendly and modern interface.

## Key Features of Frappe HR

*   **Employee Lifecycle Management:** Manage the entire employee journey, from onboarding and promotions to transfers and exit interviews.
*   **Leave and Attendance Tracking:** Configure leave policies, track attendance with geolocation, and manage leave balances efficiently.
*   **Expense Claim and Advance Management:** Streamline expense claims and employee advances with multi-level approval workflows and integration with ERPNext accounting.
*   **Performance Management:** Set and track goals, use key result areas (KRAs), and conduct performance appraisals with ease.
*   **Payroll and Taxation:** Configure salary structures, calculate taxes, generate salary slips, and manage payroll efficiently.
*   **Frappe HR Mobile App:** Access HR functionalities on the go, including leave applications and approvals, and employee profile information.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Under the Hood

*   **Frappe Framework:** The robust, full-stack web application framework written in Python and Javascript, providing the foundation for Frappe HR. ([Frappe Framework GitHub](https://github.com/frappe/frappe))
*   **Frappe UI:** A Vue-based UI library that provides a modern and responsive user interface. ([Frappe UI GitHub](https://github.com/frappe/frappe-ui))

## Production Setup

### Managed Hosting

Simplify your Frappe HR deployment with [Frappe Cloud](https://frappecloud.com). This platform offers:

*   Easy installation and setup.
*   Automated upgrades and monitoring.
*   Comprehensive maintenance and support.

<br/>
<div>
	<a href="https://frappecloud.com/hrms/signup" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>
<br/>

## Development Setup

### Docker

1.  Ensure you have Docker, docker-compose, and git installed.
2.  Clone the repository: `git clone https://github.com/frappe/hrms`
3.  Navigate to the Docker directory: `cd hrms/docker`
4.  Run `docker-compose up`
5.  Access Frappe HR in your browser at `http://localhost:8000`
6.  Use the following credentials:
    *   Username: `Administrator`
    *   Password: `admin`

### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server and keep it running
	```sh
	$ bench start
	```
2. In a separate terminal window, run the following commands
	```sh
	$ bench new-site hrms.local
	$ bench get-app erpnext
	$ bench get-app hrms
	$ bench --site hrms.local install-app hrms
	$ bench --site hrms.local add-to-hosts
	```
3.  Access the site at `http://hrms.local:8080`

## Learning and Community

*   [Frappe School](https://frappe.school) - Online courses to learn the Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive documentation for Frappe HR.
*   [User Forum](https://discuss.erpnext.com/) - Connect with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get instant help from the community.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Logo and Trademark Policy

Review the [Logo and Trademark Policy](TRADEMARK_POLICY.md) before using the Frappe HR logo.

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

Key changes and improvements:

*   **SEO-Optimized Title & Description:** The title includes the target keywords, and the description is concise and enticing.
*   **Clear Headings and Formatting:**  Organized with clear headings (e.g., "About Frappe HR," "Key Features") for readability and SEO.
*   **Bulleted Key Features:**  Makes it easy to scan and understand the core functionalities.
*   **Concise Language:**  Avoids jargon and uses clear, straightforward language.
*   **Links to GitHub:**  Added "View on GitHub" for direct access to the source code.
*   **Focus on Benefits:**  Highlights the benefits of using Frappe HR (e.g., streamline operations, empower HR).
*   **Call to Action (Implied):** Encourages users to explore the software.
*   **Contextual Links:** Linked to relevant resources within the Frappe ecosystem.
*   **Removed Redundancy:** Removed duplicate information and streamlined the text.
*   **GitHub Links**: Provided links to the code repositories where appropriate.