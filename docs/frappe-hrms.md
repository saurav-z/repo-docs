<div align="center">
  <a href="https://frappe.io/hr">
    <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
  </a>
  <h2>Frappe HR: Open Source HR & Payroll Software</h2>
  <p align="center">
    Manage your entire HR lifecycle with Frappe HR, the modern and easy-to-use open-source solution.
  </p>

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

<a href="https://trendshift.io/repositories/10972" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10972" alt="frappe%2Fhrms | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<div align="center">
  <img src=".github/hrms-hero.png" alt="Frappe HR Screenshot"/>
</div>

<div align="center">
  <a href="https://frappe.io/hr">Website</a>
  -
  <a href="https://docs.frappe.io/hr/introduction">Documentation</a>
  -  <a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## Frappe HR: Your Complete HRMS Solution

Frappe HR is a comprehensive Human Resources Management System (HRMS) designed to streamline and optimize all aspects of your employee management. With over 13 integrated modules, Frappe HR empowers businesses to efficiently manage their workforce, from onboarding to payroll and everything in between.  It's a true open-source alternative, allowing you to customize and adapt the system to your specific needs.

## Key Features of Frappe HR

*   **Employee Lifecycle Management:**  Seamlessly manage the entire employee journey, from onboarding to performance reviews and offboarding, ensuring a smooth experience for your employees.
*   **Leave and Attendance Tracking:** Configure custom leave policies, automate holiday calendars, track check-in/check-out times with geolocation, and monitor leave balances with comprehensive reporting.
*   **Expense Claims and Advances:** Simplify expense management with multi-level approval workflows and seamless integration with accounting for efficient financial control.
*   **Performance Management:**  Set and track employee goals, align them with key result areas (KRAs), and facilitate appraisal cycles with integrated tools.
*   **Payroll & Taxation:**  Create flexible salary structures, handle income tax calculations, run payroll processing, and generate detailed salary slips.
*   **Frappe HR Mobile App:** Empower your employees with on-the-go access to key HR functions, including leave applications, attendance tracking, and employee profile management.

<details open>
<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png" alt="Appraisal Screenshot"/>
	<img src=".github/hrms-requisition.png" alt="Requisition Screenshot"/>
	<img src=".github/hrms-attendance.png" alt="Attendance Screenshot"/>
	<img src=".github/hrms-salary.png" alt="Salary Screenshot"/>
	<img src=".github/hrms-pwa.png" alt="PWA Screenshot"/>
</details>

## Under the Hood: Technology Stack

*   [**Frappe Framework**](https://github.com/frappe/frappe):  A powerful, full-stack web application framework built with Python and JavaScript, providing a robust foundation for Frappe HR.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A modern, Vue.js-based UI library ensures a user-friendly and responsive interface.

## Setup and Deployment

### Managed Hosting
Consider using [Frappe Cloud](https://frappecloud.com), a managed platform that simplifies the deployment and management of Frappe applications.

<div>
	<a href="https://frappecloud.com/hrms/signup" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

### Development Setup

**Using Docker:**

1.  Ensure you have Docker, docker-compose, and git installed.
2.  Clone the repository: `git clone https://github.com/frappe/hrms`
3.  Navigate to the Docker directory: `cd hrms/docker`
4.  Run: `docker-compose up`
5.  Access the application at `http://localhost:8000` with the credentials:
    *   Username: `Administrator`
    *   Password: `admin`

**Local Setup:**

1.  Install and start Bench following the [Installation Steps](https://frappeframework.com/docs/user/en/installation).
2.  In a separate terminal:
    ```bash
    bench new-site hrms.local
    bench get-app erpnext
    bench get-app hrms
    bench --site hrms.local install-app hrms
    bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`

## Learning and Community Resources

1.  [Frappe School](https://frappe.school) - Learn from courses by the maintainers and the community.
2.  [Documentation](https://docs.frappe.io/hr) - Extensive documentation.
3.  [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://t.me/frappehr) - Get instant help.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

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
Key changes and improvements:

*   **SEO Optimization:**  Included relevant keywords like "HRMS," "HR and Payroll," "Open Source," and "Employee Management" in headings and descriptions.
*   **One-Sentence Hook:**  The introductory paragraph focuses on what Frappe HR *is* and its value proposition.
*   **Clear Headings and Structure:**  Improved readability with headings and subheadings.
*   **Bulleted Key Features:**  Presented key features in a clear, scannable bulleted list.
*   **Concise Descriptions:** Provided concise descriptions for each feature.
*   **GitHub Link:** Explicitly added a link back to the original repo.
*   **Screenshot Alt Text:** Added `alt` text to the images to help with accessibility and SEO.
*   **Managed Hosting Section:**  Expanded on the managed hosting section.
*   **Community & Learning:**  Reorganized the learning and community sections.