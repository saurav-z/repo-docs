<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HRMS Software for Modern Businesses</h2>
	<p align="center">
		<p>Manage your entire employee lifecycle with a comprehensive, open-source HR solution.</p>
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

## Frappe HR: Your Complete HR Management Solution

Frappe HR is a modern, open-source Human Resources Management System (HRMS) designed to streamline and optimize your HR processes.  Built on the robust Frappe framework, Frappe HR offers a complete suite of modules to manage your entire employee lifecycle, from onboarding to payroll and beyond.  [Explore the Frappe HR repository on GitHub](https://github.com/frappe/hrms).

## Key Features

*   **Employee Lifecycle Management:**  Effortlessly onboard employees, manage promotions and transfers, and conduct exit interviews to improve the employee experience.
*   **Leave and Attendance Tracking:** Configure custom leave policies, automate holiday calendars, and track employee attendance with geolocation features.
*   **Expense Claims and Advances:** Simplify expense reporting and employee advances with multi-level approval workflows, fully integrated with ERPNext accounting.
*   **Performance Management:** Set and track goals, align them with key result areas (KRAs), and facilitate employee self-evaluations to drive performance.
*   **Payroll and Taxation:**  Create flexible salary structures, configure tax slabs, and manage payroll with ease. Generate detailed payslips and handle off-cycle payments.
*   **Frappe HR Mobile App:**  Manage HR tasks on the go! Apply for and approve leaves, check attendance, and access employee profiles directly from your mobile device.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Built on Powerful Technology

*   **Frappe Framework:**  A full-stack web application framework written in Python and JavaScript, providing the foundation for Frappe HR's robust functionality.
*   **Frappe UI:**  A modern, Vue.js-based UI library that delivers a clean and intuitive user experience.

## Getting Started

### Production Setup

*   **Frappe Cloud:**  Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a managed hosting platform that handles installation, upgrades, monitoring, and support.

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
Install Docker and Docker Compose, then run the following commands:

```bash
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```
Access `http://localhost:8000` in your browser with the following credentials:
- Username: `Administrator`
- Password: `admin`

#### Local
1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:

    ```bash
    $ bench start
    ```

2.  In a separate terminal, run:

    ```bash
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```

3.  Access the site at `http://hrms.local:8080`

## Learn More and Get Involved

*   **Frappe School:**  Explore tutorials and courses on the Frappe Framework and ERPNext: [https://frappe.school](https://frappe.school)
*   **Documentation:**  Comprehensive documentation for Frappe HR: [https://docs.frappe.io/hr](https://docs.frappe.io/hr)
*   **User Forum:**  Connect with the community: [https://discuss.erpnext.com/](https://discuss.erpnext.com/)
*   **Telegram Group:**  Get instant help from the community: [https://t.me/frappehr](https://t.me/frappehr)

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

*   **SEO-Optimized Title and Description:** Added "Open Source HRMS Software for Modern Businesses" to the main title for better search visibility.  The introductory paragraph serves as a strong meta description.
*   **Clear and Concise Hook:**  The one-sentence hook, "Manage your entire employee lifecycle with a comprehensive, open-source HR solution,"  quickly communicates the value proposition.
*   **Keyword Integration:** The description uses relevant keywords like "HRMS," "Human Resources Management System," "open-source," and related terms.
*   **Structured Headings:**  Uses clear headings (H2, H3) to improve readability and organization.
*   **Bulleted Feature Lists:** Easy to scan key features.
*   **Internal Linking:**  Included links back to key resources like the documentation, website, and community forum.
*   **Call to action**  Links to try Frappe Cloud and the repo.
*   **Concise Language:** Removed unnecessary words and phrases.
*   **Bolded Keywords:**  Used bold text to highlight key features and important terms.
*   **Improved Content:**  Improved the overall flow and readability of the text.
*   **Stronger Emphasis on Open Source:** Explicitly mentions the open-source nature as a key selling point.
*   **Clearer Instructions** Improved the setup instructions for clarity.