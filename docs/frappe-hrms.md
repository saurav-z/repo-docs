<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HR and Payroll Software</h2>
	<p align="center">
		<p>Manage your entire employee lifecycle with Frappe HR, a modern, open-source HR and Payroll solution.</p>
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
    -  <a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## Frappe HR: Your Complete HRMS Solution

Frappe HR is a powerful, open-source Human Resource Management System (HRMS) designed to streamline and automate your HR processes.  From employee management to payroll, Frappe HR offers a comprehensive suite of modules to drive excellence within your company.  Built on the robust Frappe Framework, Frappe HR provides a modern and user-friendly experience.

## Key Features of Frappe HR

*   **Employee Lifecycle Management:**  Onboard, manage, and offboard employees seamlessly.  Track promotions, transfers, and performance feedback.
*   **Leave and Attendance Tracking:**  Configure custom leave policies, manage time off requests, and monitor attendance with geolocation check-ins.
*   **Expense Claims and Advances:**  Simplify expense reporting with multi-level approval workflows and seamless ERPNext accounting integration.
*   **Performance Management:**  Define goals, align them with key result areas (KRAs), and streamline performance appraisal cycles.
*   **Payroll & Taxation:**  Create salary structures, manage income tax, run payroll, and generate salary slips with detailed income breakdowns.
*   **Frappe HR Mobile App:**  Access essential HR functions on the go, including leave applications, approvals, and employee profile information.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

## Under the Hood

Frappe HR leverages powerful underlying technologies:

*   **Frappe Framework:**  A full-stack Python and JavaScript web application framework, providing a strong foundation for building web applications.
*   **Frappe UI:** A modern, Vue-based UI library for a user-friendly interface.

## Production Setup

### Managed Hosting

For simplified deployment, explore [Frappe Cloud](https://frappecloud.com), a platform for hosting Frappe applications, with installation, upgrades, and maintenance provided.

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

1.  **Prerequisites:** Ensure Docker, docker-compose, and Git are installed.
2.  **Clone Repository:** `git clone https://github.com/frappe/hrms`
3.  **Navigate:** `cd hrms/docker`
4.  **Run:** `docker-compose up`

Access the HR instance at `http://localhost:8000` with:

-   Username: `Administrator`
-   Password: `admin`

### Local

1.  Set up bench following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server: `$ bench start`
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

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Extensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get instant help from the user community.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Logo and Trademark Policy

Review the [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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

*   **Clear Title and Hook:**  The title includes relevant keywords ("HR" and "Payroll Software"), and the one-sentence hook grabs attention.
*   **Keyword Optimization:**  Keywords are incorporated naturally throughout the text (e.g., "HRMS," "Human Resource Management System," "Payroll," "Employee Lifecycle").
*   **Bulleted Key Features:**  Easy-to-read bullet points highlight the core functionalities, improving readability and scannability.
*   **Structured Headings:**  Using `<h2>` and other heading levels to structure the information makes it easier for users (and search engines) to understand the content's organization.
*   **Internal and External Links:** Links to the original GitHub repo, documentation, website, and community resources enhance user engagement and SEO.
*   **Concise and Informative Content:** The descriptions are clear, concise, and highlight the benefits of using Frappe HR.
*   **Call to Action:** Encourages exploration through links.
*   **Screenshots:** Screenshots have been maintained within the `<details>` tag.
*   **Alt tags:** Alt tags are used for images, aiding in accessibility and SEO.
*   **GitHub link:** Added a direct link back to the original GitHub repository.
*   **Modern Tone:**  Maintains a professional and approachable tone.