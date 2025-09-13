<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HRMS Software</h2>
	<p align="center">
		<p>Manage your employees with ease using Frappe HR, a modern and comprehensive open-source HR and Payroll solution.</p>
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

Frappe HR is a complete, open-source Human Resources Management System (HRMS) designed to streamline all aspects of your HR operations. From employee onboarding to payroll, Frappe HR provides a modern, user-friendly solution to manage your workforce effectively.  This software is a complete HRMS solution with over 13 different modules, including Employee Management, Onboarding, Leaves, Payroll, and Taxation.

## Key Features

*   **Employee Lifecycle Management:** Easily manage the entire employee journey, from onboarding and promotions to feedback and exit interviews.
*   **Leave and Attendance Tracking:** Configure flexible leave policies, automatically incorporate regional holidays, track employee check-in/check-out with geolocation, and generate insightful reports.
*   **Expense Claims and Advances:** Simplify expense reporting with multi-level approval workflows and seamless integration with ERPNext accounting.
*   **Performance Management:** Track employee goals, align them with key result areas (KRAs), and manage appraisal cycles efficiently.
*   **Payroll and Taxation:** Configure salary structures, manage income tax, run payroll, handle additional payments, and generate detailed salary slips.
*   **Mobile App:** Access key HR features on the go with the Frappe HR mobile app, including leave applications, approvals, and employee profile access.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

## Technical Underpinnings

Frappe HR is built upon robust open-source technologies:

*   **Frappe Framework:** A full-stack web application framework (Python and Javascript) providing the foundation for Frappe HR.
*   **Frappe UI:** A Vue.js-based UI library for a modern and intuitive user interface.

## Deployment Options

### Managed Hosting

For a hassle-free deployment experience, consider [Frappe Cloud](https://frappecloud.com), a platform that handles installation, setup, upgrades, monitoring, and support.

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
2.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```
3.  Access the application at `http://localhost:8000` with the following credentials:
    *   Username: `Administrator`
    *   Password: `admin`

#### Local Installation

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation). Start the server:
    ```sh
    $ bench start
    ```
2.  In a separate terminal:
    ```sh
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`

## Resources & Community

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get instant help.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Legal

*   [Logo and Trademark Policy](TRADEMARK_POLICY.md)

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

*   **Concise and SEO-Friendly Title:** "Frappe HR: Open Source HRMS Software" uses relevant keywords.
*   **One-Sentence Hook:** Immediately highlights the core value proposition.
*   **Clear Headings:** Organized content for readability and searchability.
*   **Bulleted Key Features:** Easy for users to scan and understand key benefits.  Added relevant keywords to feature descriptions.
*   **Focus on User Benefits:** The descriptions of features are more focused on *what* users get (e.g., "Easily manage the entire employee journey...") rather than just *what* the software *does*.
*   **Strategic Keyword Use:** The text incorporates relevant keywords like "HRMS," "open source," "HR and Payroll," "employee management," etc., naturally.
*   **Clear Call to Actions:** Links to website, documentation, and a direct link back to the GitHub repo.
*   **Deployment Options emphasized.**  Made both Cloud and Development set up a bit clearer.
*   **Concise descriptions.** The motivation was removed as it's not as important to a user looking for a solution.
*   **Improved Readability:**  Using bolding and bullet points for increased scannability.
*   **View on GitHub link** Added for easier navigation.