<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open-Source HR & Payroll Software</h2>
	<p align="center">
		Simplify your HR processes and empower your workforce with Frappe HR, the modern, open-source HRMS solution.
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

## Frappe HR: Your Complete HRMS Solution

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline all aspects of HR.  With over 13 modules, it's a one-stop solution for managing your employees and driving operational excellence.  

**[Explore Frappe HR on GitHub](https://github.com/frappe/hrms)**

### Key Features:

*   **Employee Lifecycle Management:**  Handle everything from onboarding and promotions to transfers and exit interviews, making employee management efficient.
*   **Leave and Attendance Tracking:** Configure leave policies, automate holiday calendars, use geolocation check-in/out, and generate attendance reports.
*   **Expense Claims and Advances:**  Manage employee advances and expenses with multi-level approval workflows and seamless integration with ERPNext accounting.
*   **Performance Management:** Track employee goals, align them with key result areas (KRAs), enable self-evaluations, and simplify appraisal cycles.
*   **Payroll & Taxation:** Create flexible salary structures, configure tax slabs, process payroll, manage additional payments, and generate detailed salary slips.
*   **Mobile App:**  Access Frappe HR on the go with the mobile app, enabling leave applications, approvals, and employee profile access.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Technology Behind Frappe HR

*   **Frappe Framework:** A full-stack web application framework built on Python and Javascript, providing a robust foundation for the application.
*   **Frappe UI:** A modern, Vue-based UI library that delivers a user-friendly interface.

## Production Setup

### Managed Hosting

For ease of use and peace of mind, consider [Frappe Cloud](https://frappecloud.com). It simplifies the deployment, management, and maintenance of your Frappe HR instance.

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

Set up your development environment with Docker using the following commands:

```
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```

Access the application at `http://localhost:8000` using the credentials:

*   Username: `Administrator`
*   Password: `admin`

### Local Setup

Follow these steps to set up a local development environment:

1.  Set up bench: [Installation Steps](https://frappeframework.com/docs/user/en/installation) and run the server:
    ```sh
    $ bench start
    ```
2.  In a separate terminal window:
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
*   [Documentation](https://docs.frappe.io/hr): Comprehensive documentation for Frappe HR.
*   [User Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr): Get instant community support.

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
```
Key improvements and SEO considerations:

*   **Clear, concise, keyword-rich title:**  "Frappe HR: Open-Source HR & Payroll Software" directly targets the search terms.
*   **Strong one-sentence hook:**  "Simplify your HR processes and empower your workforce with Frappe HR, the modern, open-source HRMS solution." Immediately conveys the value proposition.
*   **Strategic use of keywords:**  Incorporates "HRMS," "HR," "Payroll," "Open-Source," and other relevant terms naturally throughout the text.
*   **Clear headings and subheadings:**  Improves readability and helps search engines understand the content structure.
*   **Bulleted key features:**  Easy to scan and highlights the main benefits.
*   **Link back to the original repo (as requested).**
*   **Emphasis on benefits:** Focuses on what users *gain* (e.g., streamlined processes, empowered workforce).
*   **Call to Action:** Provides the link to the GitHub repo at the beginning.
*   **Organized content:** Improves readability and user experience.
*   **Alt tags on images:** These are important for accessibility and SEO, but were not provided in the original.  I left them in the code where the images are.