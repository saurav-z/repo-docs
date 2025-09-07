<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open-Source HR and Payroll Software</h2>
	<p align="center">
		<p>Manage your entire employee lifecycle with a modern, open-source HRMS solution.</p>
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
	<a href="https://github.com/frappe/hrms">GitHub Repository</a>
</div>

---

## Frappe HR: Your Complete Open-Source HRMS Solution

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline and automate your HR processes. Built with modern technology, it's easy to use and packed with features to manage your entire employee lifecycle.

### Key Features:

*   **Employee Lifecycle Management:** From onboarding and promotions to performance reviews and offboarding, Frappe HR simplifies every stage of the employee journey.
*   **Leave and Attendance Tracking:** Manage leave policies, track attendance with geolocation, and generate insightful reports.
*   **Expense Claims and Advances:**  Process employee advances and expense claims with multi-level approval workflows, integrated with ERPNext accounting.
*   **Performance Management:** Set goals, align them with Key Result Areas (KRAs), conduct self-evaluations, and manage appraisal cycles efficiently.
*   **Payroll and Taxation:**  Create salary structures, configure tax slabs, run payroll, and generate salary slips.
*   **Mobile App:** Empower your employees with the Frappe HR mobile app for leave requests, attendance tracking, and accessing employee profiles on the go.

<details open>
<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Technology Under the Hood:

*   **[Frappe Framework](https://github.com/frappe/frappe):** A powerful full-stack web application framework built with Python and JavaScript, providing a robust foundation for the HRMS.
*   **[Frappe UI](https://github.com/frappe/frappe-ui):**  A Vue-based UI library that provides a modern user interface for a seamless user experience.

---

## Deployment Options

### Managed Hosting

For a hassle-free experience, consider [Frappe Cloud](https://frappecloud.com), a platform to host Frappe applications with ease. It handles installation, upgrades, monitoring, and support.

<div>
	<a href="https://frappecloud.com/hrms/signup" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

### Development Setup
**1. Docker (Recommended)**

1.  Ensure you have Docker, docker-compose, and git installed. Refer to the [Docker documentation](https://docs.docker.com/).
2.  Run the following commands:
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```
3.  Access Frappe HR at `http://localhost:8000` using the credentials:
    *   Username: `Administrator`
    *   Password: `admin`

**2. Local Setup**
1.  Install Bench: Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation).
2.  Start the server:
    ```bash
    bench start
    ```
3.  In a separate terminal:
    ```bash
    bench new-site hrms.local
    bench get-app erpnext
    bench get-app hrms
    bench --site hrms.local install-app hrms
    bench --site hrms.local add-to-hosts
    ```
4.  Access the site at `http://hrms.local:8080`

---

## Learn More and Get Involved

*   [Frappe School](https://frappe.school) - Learn the Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Detailed documentation for Frappe HR.
*   [User Forum](https://discuss.erpnext.com/) - Connect with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get quick help from other users.

---

## Contribute

We welcome contributions!  Please review the following:

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

---

## Trademark

See our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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

*   **SEO Title:**  The main heading is improved with the inclusion of "Open-Source HR and Payroll Software" to help with search ranking.
*   **One-Sentence Hook:**  A concise, benefit-driven opening sentence.
*   **Keyword Optimization:**  The text is infused with relevant keywords (HRMS, Human Resources, Payroll, Employee Management, Open Source).
*   **Clear Headings:**  Uses clear, descriptive headings and subheadings for readability and SEO.
*   **Bulleted Key Features:**  Easy-to-scan bullet points highlight the main benefits and features.
*   **"Why Frappe HR" implicitly conveyed:** The existing "Motivation" section is implied through the value proposition of the product rather than stating the reasons to use it.
*   **Call to Action (Implied):**  Links to the website, documentation, and GitHub repository encourage exploration.
*   **Concise Summary:** The text is streamlined to quickly convey the value proposition.
*   **Link Back to Original Repo:**  The GitHub Repository link has been added.
*   **Improved Formatting:** The structure is more organized, which is crucial for both readability and search engine optimization.

This revised README is more informative, user-friendly, and optimized for search engines. It should improve discoverability and attract users interested in a complete HRMS solution.