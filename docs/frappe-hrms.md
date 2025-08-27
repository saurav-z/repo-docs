<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open-Source HRMS Software</h2>
	<p align="center">
		<p><b>Manage your entire employee lifecycle with Frappe HR, a modern, open-source HR and payroll solution built for efficiency and ease of use.</b></p>
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

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline all your HR processes. Built on the robust Frappe Framework, Frappe HR offers a modern and intuitive interface, making HR management efficient and user-friendly.  It's a complete HRMS solution with over 13 different modules right from Employee Management, Onboarding, Leaves, to Payroll, Taxation, and more!

**[Explore the Frappe HR GitHub Repository](https://github.com/frappe/hrms)**

## Key Features

*   **Employee Lifecycle Management:** Manage the entire employee journey, from onboarding and promotions to transfers and exit interviews, enhancing the employee experience.
*   **Leave and Attendance Tracking:** Configure custom leave policies, automatically pull in regional holidays, utilize geolocation for check-in/check-out, and generate detailed leave and attendance reports.
*   **Expense Claims & Advances:** Simplify expense management with employee advances, claim submissions, and multi-level approval workflows, with seamless integration with ERPNext accounting.
*   **Performance Management:** Set and track employee goals, align them with Key Result Areas (KRAs), enable self-evaluations, and simplify appraisal cycles.
*   **Payroll & Taxation:** Create flexible salary structures, configure tax slabs, process payroll accurately, handle additional payments, and provide clear salary slips.
*   **Frappe HR Mobile App:** Empower your employees with a mobile app for leave applications and approvals, check-ins/check-outs, and easy access to employee profiles.

<details open>
<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

## Tech Stack & Under the Hood

*   **[Frappe Framework](https://github.com/frappe/frappe):** A powerful full-stack web application framework built with Python and Javascript, providing a solid foundation.
*   **[Frappe UI](https://github.com/frappe/frappe-ui):** A modern, Vue-based UI library that offers a user-friendly and responsive interface.

## Getting Started

### Managed Hosting

For a hassle-free experience, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform that simplifies the deployment, management, and maintenance of Frappe applications.

<div>
	<a href="https://frappecloud.com/hrms/signup" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

### Development Setup with Docker

1.  Ensure you have Docker, docker-compose, and Git installed. Refer to the [Docker documentation](https://docs.docker.com/) for setup instructions.
2.  Clone the repository and navigate to the `docker` directory:
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    ```
3.  Run `docker-compose up` to start the application.
4.  Access Frappe HR at `http://localhost:8000` using the credentials:
    *   Username: `Administrator`
    *   Password: `admin`

### Local Development

1.  Install and start the Bench server by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and run:
    ```sh
    $ bench start
    ```
2.  In a separate terminal window, run the following commands:
    ```sh
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`

## Learning and Community

*   [Frappe School](https://frappe.school): Learn the Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr): Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr): Get instant help from fellow users.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

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
```
Key improvements and SEO considerations:

*   **Clear Title:** The title "Frappe HR: Open-Source HRMS Software" is optimized with keywords.
*   **One-Sentence Hook:**  The first paragraph provides a concise and engaging introduction.
*   **Keyword Optimization:** Keywords like "HRMS," "open-source," "HR," and "payroll" are integrated throughout.
*   **Structured Headings:**  Headings are used to organize content and improve readability.
*   **Bulleted Key Features:** This is a common SEO practice for emphasizing benefits.
*   **Actionable Language:**  The text uses active verbs and encourages engagement (e.g., "Explore," "Manage").
*   **Internal Links:**  Links to the GitHub repository, documentation, and other resources are included.
*   **Concise Descriptions:**  Each section provides a focused summary of the software's features.
*   **Clear Call to Action:** Promotes the "Get Started" section, making it easy for users to start.
*   **Mobile App is included** This is a key feature.
*   **Strong Summary:** Provides a clear overview of the product and its benefits.