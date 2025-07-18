<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HR and Payroll Software</h2>
	<p align="center">
		<p>Modernize your HR processes with Frappe HR, a comprehensive and easy-to-use solution.</p>
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

## Frappe HR: Your Complete Open-Source HRMS Solution

Frappe HR is a powerful, open-source Human Resources Management System (HRMS) designed to streamline your HR operations. Built on the Frappe Framework, it offers a comprehensive suite of modules to manage the entire employee lifecycle, from onboarding to payroll and beyond.  [Explore Frappe HR on GitHub](https://github.com/frappe/hrms) to learn more.

## Key Features

Frappe HR empowers businesses with a robust set of features, including:

*   **Employee Lifecycle Management**: Simplify employee onboarding, manage promotions, track transfers, and conduct exit interviews to enhance the employee experience.
*   **Leave and Attendance Tracking**: Configure custom leave policies, automate holiday calendars, use geolocation for check-in/check-out, and generate attendance reports.
*   **Expense Claims and Advances**: Manage employee advances, claim expenses with multi-level approval workflows, and integrate seamlessly with ERPNext accounting.
*   **Performance Management**: Set and track goals, align them with key result areas (KRAs), facilitate employee self-evaluations, and streamline appraisal cycles.
*   **Payroll & Taxation**: Create flexible salary structures, configure income tax slabs, process standard and off-cycle payroll, and generate detailed salary slips.
*   **Mobile App**:  Stay connected with the Frappe HR mobile app for on-the-go leave applications, approvals, and employee profile access.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Under the Hood

*   **Frappe Framework**:  A full-stack web application framework (Python and Javascript) providing a robust foundation for web applications.
*   **Frappe UI**: A Vue-based UI library for a modern user interface.

## Production Setup

### Managed Hosting

For a hassle-free experience, try [Frappe Cloud](https://frappecloud.com). This open-source platform simplifies hosting, management, and maintenance of your Frappe applications.

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

Requires Docker, docker-compose, and git.

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```

2.  Access your HR instance at `http://localhost:8000`

    *   Username: `Administrator`
    *   Password: `admin`

### Local

1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up Bench and start the server:
    ```bash
    bench start
    ```

2.  In a separate terminal:
    ```bash
    bench new-site hrms.local
    bench get-app erpnext
    bench get-app hrms
    bench --site hrms.local install-app hrms
    bench --site hrms.local add-to-hosts
    ```

3.  Access the site at `http://hrms.local:8080`

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
2.  [Documentation](https://docs.frappe.io/hr) - Extensive documentation for Frappe HR.
3.  [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://t.me/frappehr) - Get instant help from the community.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Logo and Trademark Policy

Please review the [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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

*   **SEO Optimization:** Included keywords like "open source HR," "HRMS," "HR and Payroll Software" in headings and descriptions.
*   **One-Sentence Hook:**  The opening sentence immediately grabs attention: "Modernize your HR processes with Frappe HR, a comprehensive and easy-to-use solution."
*   **Clear Headings:**  Used clear, concise headings for each section.
*   **Bulleted Key Features:**  Organized key features into bullet points for easy readability.
*   **Summarized Content:**  Condensed information while retaining key details.
*   **GitHub Link:**  Added a direct link back to the GitHub repository at the top and within the introductory paragraph.
*   **Call to Action (CTA):**  Subtly encourages the user to "Explore Frappe HR on GitHub."
*   **Structure and Formatting:** Improved the overall readability and visual appeal using Markdown formatting.
*   **Removed Redundancy:**  Streamlined explanations and removed repeated information.
*   **Concise Development Setup:** Refined the development setup instructions.
*   **Community and Support Highlight:** Emphasized the community and learning resources.