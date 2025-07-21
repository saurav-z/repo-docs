<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HR & Payroll Software</h2>
	<p align="center">
		<p>Manage your entire employee lifecycle with Frappe HR, the modern, open-source HRMS solution.</p>
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
	-
	<a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## Frappe HR: Your Comprehensive HRMS Solution

Frappe HR is a powerful, open-source Human Resources Management System designed to streamline and automate your HR processes. This robust solution offers over 13 modules, covering everything from employee management and onboarding to payroll, taxation, and more.  **[View the source code on GitHub](https://github.com/frappe/hrms)**.

## Key Features of Frappe HR

*   **Employee Lifecycle Management:** Simplify employee onboarding, manage promotions and transfers, and conduct exit interviews with ease.
*   **Leave and Attendance Tracking:** Configure flexible leave policies, integrate regional holidays, utilize geolocation check-in/out, and generate attendance reports.
*   **Expense Claims & Advances:** Manage employee advances, claim expenses, and configure multi-level approval workflows, seamlessly integrated with ERPNext accounting.
*   **Performance Management:** Set and track goals, align them with key result areas (KRAs), enable employee self-evaluations, and streamline appraisal cycles.
*   **Payroll & Taxation:** Create custom salary structures, configure income tax slabs, run payroll, handle additional payments, and provide detailed salary slips.
*   **Frappe HR Mobile App:** Access key HR functionalities on the go, including leave requests and approvals, check-in/out, and employee profile management.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Built on Powerful Technologies

*   **Frappe Framework:** A full-stack web application framework (Python & Javascript) providing the foundation for robust and scalable web applications.
*   **Frappe UI:** A modern, Vue-based UI library to create a user-friendly interface.

## Production Setup

### Managed Hosting

For a hassle-free experience, consider [Frappe Cloud](https://frappecloud.com).  This platform simplifies hosting, setup, upgrades, monitoring, maintenance, and support for your Frappe HR deployment.

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

1.  Install Docker, docker-compose, and Git on your machine.
2.  Run these commands:

```bash
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```
3.  Access `http://localhost:8000` in your browser.
4.  Login with:
    -   Username: `Administrator`
    -   Password: `admin`

### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server.
    ```bash
    $ bench start
    ```
2.  In a separate terminal:
    ```bash
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`

## Learn More and Get Involved

1.  [Frappe School](https://frappe.school): Learn Frappe Framework and ERPNext.
2.  [Documentation](https://docs.frappe.io/hr): Comprehensive Frappe HR documentation.
3.  [User Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
4.  [Telegram Group](https://t.me/frappehr): Get instant support from the community.

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
Key changes and improvements:

*   **SEO Optimization:**  Included relevant keywords ("HRMS," "HR and Payroll Software," "Open Source") in the title and throughout the document.
*   **One-Sentence Hook:** The initial sentence provides a clear value proposition.
*   **Clear Headings:**  Used headings for better organization and readability.
*   **Bulleted Key Features:**  Easy to scan and highlights the key benefits.
*   **Concise Language:**  Improved clarity and reduced unnecessary words.
*   **Call to Action:** Added a link to "View on GitHub" at the top.
*   **Community Links:** Reorganized the links to make it easy to navigate.
*   **Added GitHub Link:** Provided a link to the original repo.