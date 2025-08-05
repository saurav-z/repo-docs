<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HR and Payroll Software</h2>
	<p align="center">
		<p>A comprehensive HRMS solution designed to streamline and optimize your HR processes.</p>
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
    - 	<a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## About Frappe HR

Frappe HR is a powerful, open-source Human Resources Management System (HRMS) built to empower your organization. It's a complete solution with over 13 modules covering everything from employee management and onboarding to leave management, payroll, and taxation.  Developed by the team at Frappe, it offers a modern, user-friendly experience.

## Key Features of Frappe HR

*   **Employee Lifecycle Management**:  Manage employees from onboarding through promotions, transfers, and exit interviews, streamlining the entire employee journey.
*   **Leave and Attendance Tracking**: Configure leave policies, easily manage regional holidays, track check-in/check-out with geolocation, and monitor leave balances and attendance.
*   **Expense Claims and Advances**:  Manage employee advances, expense claims, and configure multi-level approval workflows with seamless integration with ERPNext accounting.
*   **Performance Management**: Track employee goals, align goals with key result areas (KRAs), facilitate self-evaluations, and simplify appraisal cycles.
*   **Payroll & Taxation**: Create salary structures, configure income tax slabs, process payroll, manage additional salaries and off-cycle payments, and view income breakdowns on salary slips.
*   **Mobile App**: Access key HR functions on the go with the Frappe HR mobile app, including leave requests/approvals, check-in/check-out, and employee profile access.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Under the Hood

-   **Frappe Framework**: [Frappe Framework](https://github.com/frappe/frappe) powers Frappe HR, providing a robust foundation with a full-stack web application framework written in Python and Javascript.
-   **Frappe UI**: [Frappe UI](https://github.com/frappe/frappe-ui) ensures a modern and intuitive user interface based on Vue.js.

## Get Started

### Production Setup

Easily host your Frappe HR instance with [Frappe Cloud](https://frappecloud.com), an easy-to-use open-source platform. It handles installation, setup, upgrades, monitoring, and maintenance.

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

1.  Ensure you have Docker, docker-compose, and Git installed.
2.  Clone the repository: `git clone https://github.com/frappe/hrms`
3.  Navigate to the Docker directory: `cd hrms/docker`
4.  Start the containers: `docker-compose up`
5.  Access the application at `http://localhost:8000`.
6.  Login using:  Username: `Administrator`, Password: `admin`

#### Local

1.  Follow the [Frappe Framework Installation Steps](https://frappeframework.com/docs/user/en/installation).
2.  Start the bench server: `$ bench start`
3.  In a separate terminal, run the following commands:
    ```sh
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
4.  Access the site at `http://hrms.local:8080`

## Resources & Community

*   [Frappe School](https://frappe.school) - Learn the Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the community.
*   [Telegram Group](https://t.me/frappehr) - Get instant help.

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
Key improvements and SEO considerations:

*   **Clear, Concise Headline:**  Added a keyword-rich headline: "Frappe HR: Open Source HR and Payroll Software".
*   **SEO-Friendly Summary:** The opening paragraph provides a clear value proposition and includes relevant keywords.
*   **Targeted Keywords:** Incorporated keywords throughout (HR, HRMS, Payroll, Open Source, Employee Management, etc.).
*   **Bulleted Key Features:**  Uses bullet points for easy readability and highlights the main benefits.
*   **Clear Headings:** Structured with clear headings and subheadings for better organization and scanning.
*   **Call to Action (Implied):** Encourages users to try the software.
*   **Internal Linking:** Includes links to other resources within the project and related projects.
*   **External Links (Do-Follow):** Links to relevant resources such as the documentation and the Frappe Cloud.
*   **GitHub Link Added:** Added a link back to the original repo in the intro.
*   **Simplified Development Setup Section:**  Streamlined the Docker and Local setup instructions.
*   **Community Focus:**  Highlights the resources available for learning and getting help.
*   **Removed redundant information**.