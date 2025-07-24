<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HRMS Software</h2>
	<p align="center">
		<p>Manage your entire employee lifecycle with Frappe HR, a modern and easy-to-use HR and Payroll solution.</p>
	</p>

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

</div>

<div align="center">
	<img src=".github/hrms-hero.png"/>
</div>

<div align="center">
	<a href="https://frappe.io/hr">Website</a>
	-
	<a href="https://docs.frappe.io/hr/introduction">Documentation</a>
</div>

## Frappe HR: Your Complete Open-Source HRMS Solution

Frappe HR empowers businesses with a comprehensive, open-source Human Resource Management System (HRMS).  From employee management to payroll processing, Frappe HR provides a complete suite of tools to streamline your HR operations.  Explore the official repository on [GitHub](https://github.com/frappe/hrms).

## Key Features

*   **Employee Lifecycle Management:** Simplify employee onboarding, manage promotions and transfers, and document performance through exit interviews.
*   **Leave and Attendance Tracking:**  Configure flexible leave policies, automate holiday schedules, and monitor employee attendance with geolocation features and detailed reporting.
*   **Expense Claims and Advances:** Manage employee advances, claim expenses, configure multi-level approval workflows, all this with seamless integration with ERPNext accounting.
*   **Performance Management:** Track goals, align them with key result areas (KRAs), and enable employee self-evaluation. Streamline your appraisal cycles for improved performance.
*   **Payroll & Taxation:**  Create salary structures, configure income tax slabs, run payroll, handle additional salaries and off-cycle payments, and generate detailed salary slips.
*   **Mobile Accessibility:**  Manage your HR tasks on the go with the Frappe HR mobile app, enabling leave applications, approvals, and employee profile access.

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

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.  Frappe Cloud handles installation, upgrades, monitoring, and support.

<div>
	<a href="https://frappecloud.com/hrms/signup" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

## Development setup

### Docker
You need Docker, docker-compose and git setup on your machine. Refer [Docker documentation](https://docs.docker.com/). After that, run the following commands:
```
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```

Wait for some time until the setup script creates a site. After that you can access `http://localhost:8000` in your browser and the login screen for HR should show up.

Use the following credentials to log in:

- Username: `Administrator`
- Password: `admin`

### Local

1. Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server and keep it running
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
3. You can access the site at `http://hrms.local:8080`

## Learning and Community

1. [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
2. [Documentation](https://docs.frappe.io/hr) - Extensive documentation for Frappe HR.
3. [User Forum](https://discuss.erpnext.com/) - Engage with the community of ERPNext users and service providers.
4. [Telegram Group](https://t.me/frappehr) - Get instant help from the community of users.


## Contributing

1. [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
1. [Report Security Vulnerabilities](https://erpnext.com/security)
1. [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)


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

*   **Clear, Concise Title & Description:** The title and first paragraph now immediately explain what Frappe HR is and its core value proposition. Keywords like "open-source HRMS," "HR and Payroll," and "Employee Lifecycle" are included.
*   **Keyword-Rich Headings:** Headings use relevant keywords to improve search engine ranking.
*   **Bulleted Key Features:**  Key features are clearly presented in a bulleted list for easy scanning and understanding.  Each bullet point includes relevant keywords.
*   **Link Back to Original Repo:** The text includes the link back to the original repository, to ensure attribution.
*   **Improved Readability:** Formatting and spacing are improved for better readability.
*   **Target Audience:**  Focuses on the benefits for businesses.
*   **Call to Action:** The summary entices the reader to explore the features.
*   **Overall SEO:** The updated README is designed to attract users searching for open-source HR solutions.