<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HR & Payroll Software</h2>
	<p align="center">
		<p>Empower your workforce with Frappe HR, a modern, open-source HR and payroll solution designed for efficiency and ease of use.</p>
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

## About Frappe HR

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline HR processes. This powerful solution provides over 13 modules, addressing everything from employee management and onboarding to payroll, leave management, and performance tracking. Built for modern businesses, Frappe HR offers a user-friendly interface and robust features to optimize HR operations and drive excellence within your company.

## Key Features

*   **Employee Lifecycle Management:** Manage employees throughout their entire journey, from onboarding and promotions to performance reviews and exit interviews, simplifying HR operations.
*   **Comprehensive Leave and Attendance Tracking:** Configure flexible leave policies, integrate regional holidays, track employee attendance with geolocation, and easily manage leave balances.
*   **Expense Claims and Advances:** Simplify expense management with employee advances, expense claim submissions, and multi-level approval workflows, seamlessly integrated with ERPNext accounting.
*   **Performance Management:** Set and track employee goals, align them with key result areas (KRAs), enable self-evaluations, and streamline appraisal cycles.
*   **Robust Payroll & Taxation:** Easily create salary structures, configure income tax slabs, run payroll, handle off-cycle payments, and view detailed income breakdowns on salary slips.
*   **Mobile Accessibility:** Access HR functions on the go with the Frappe HR mobile app, including leave applications, approvals, and employee profile access.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

## Under the Hood

Frappe HR is built on the solid foundation of:

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework written in Python and Javascript. It provides a robust foundation for building web applications, including a database abstraction layer, user authentication, and a REST API.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library, to provide a modern user interface. The Frappe UI library provides a variety of components that can be used to build single-page applications on top of the Frappe Framework.

## Getting Started

### Production Setup

Consider [Frappe Cloud](https://frappecloud.com) for easy managed hosting, which handles installation, upgrades, monitoring, and support.

<div>
	<a href="https://frappecloud.com/hrms/signup" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

### Development Setup

#### Using Docker

1.  Ensure you have Docker, docker-compose, and git installed.
2.  Clone the repository: `git clone https://github.com/frappe/hrms`
3.  Navigate to the Docker directory: `cd hrms/docker`
4.  Run: `docker-compose up`
5.  Access the HR application at `http://localhost:8000` using the credentials:
    *   Username: `Administrator`
    *   Password: `admin`

#### Local Setup

1.  Install and start the Frappe Bench:  follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server and keep it running `bench start`
2.  Open a new terminal window and run:
    *   `bench new-site hrms.local`
    *   `bench get-app erpnext`
    *   `bench get-app hrms`
    *   `bench --site hrms.local install-app hrms`
    *   `bench --site hrms.local add-to-hosts`
3.  Access the site at `http://hrms.local:8080`

## Learn & Contribute

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Detailed Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/) - Connect with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get instant help.

## Contribute

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
```
Key improvements and SEO considerations:

*   **Clear, Keyword-Rich Title:** The title is improved to include "Open Source HR & Payroll Software".  This helps search engines understand what the project is.
*   **One-Sentence Hook:** A concise opening sentence highlights the core value proposition.
*   **Improved Headings:** Uses clear and descriptive headings (e.g., "About Frappe HR", "Key Features", "Getting Started").
*   **Bulleted Key Features:**  Makes it easy to scan and understand the core functionalities.  The bullet points are also more descriptive and keyword-rich.
*   **Added a Link Back to the Repo:**  This provides a clear call to action and helps with discoverability.
*   **SEO-Friendly Descriptions:** The text is written to be more engaging and includes relevant keywords to improve search engine rankings.
*   **Concise and Organized Structure:** The README is organized for easy readability and navigation.
*   **Clear Call to Action:**  Provides links to relevant resources (website, documentation, community).
*   **More Descriptive Language:** Uses words like "comprehensive", "streamline", and "powerful" to make the content more appealing.
*   **Links to Related Projects and Information:** Links to the underlying Frappe Framework and Frappe UI, demonstrating the project's technical foundation.
*   **Optimized for Search:** Uses keywords like "HRMS", "Payroll", "Open Source", and other related terms throughout the text to improve search engine optimization.
*   **Includes Frappe Cloud mention** - Provides information on managed hosting options.

This improved README provides a more compelling introduction to Frappe HR and is designed to attract potential users and contributors by being more SEO-friendly and user-friendly.