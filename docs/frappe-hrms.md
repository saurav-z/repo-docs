<div align="center">
  <a href="https://frappe.io/hr">
    <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
  </a>
  <h2>Frappe HR: Open-Source HR and Payroll Software</h2>
  <p>
    <b>Manage your entire employee lifecycle with Frappe HR, a modern, easy-to-use, and open-source HRMS solution designed for businesses of all sizes.</b>
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
  -
  <a href="https://github.com/frappe/hrms">GitHub Repository</a>
</div>

## About Frappe HR

Frappe HR is a comprehensive HRMS (Human Resource Management System) built to streamline and optimize your HR processes. It's open-source, meaning you have complete control and freedom to customize it to your needs. Developed by the Frappe team, it offers a robust suite of modules, perfect for modern businesses.  Frappe HR allows you to drive excellence within the company. It's a complete HRMS solution with over 13 different modules right from Employee Management, Onboarding, Leaves, to Payroll, Taxation, and more!

### Why Choose Frappe HR?

Frappe HR was created to address the need for a truly open-source HR and Payroll solution. As the Frappe team grew, the need for a comprehensive, customizable HR platform became evident. Version 14 onwards, as the modules became more mature, Frappe HR was created as a separate product.

## Key Features of Frappe HR

*   **Employee Lifecycle Management:** Handle the entire employee journey, from onboarding and promotions to transfers and exit interviews, creating a smoother experience for your employees.
*   **Leave and Attendance Tracking:** Configure flexible leave policies, automatically pull regional holidays, use geolocation check-in/check-out, and generate attendance reports.
*   **Expense Claims and Advances:** Manage employee advances, expense claims, and multi-level approval workflows, seamlessly integrated with ERPNext accounting.
*   **Performance Management:** Track employee goals, align them with KRAs (Key Result Areas), facilitate self-evaluations, and simplify appraisal cycles.
*   **Payroll & Taxation:** Create flexible salary structures, configure income tax slabs, run payroll processes, manage additional salaries, view income breakdowns, and generate salary slips.
*   **Frappe HR Mobile App:** Empower your team to apply for and approve leaves, check in and out, and access employee profiles on the go.

<details open>
<summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png"/>
    <img src=".github/hrms-requisition.png"/>
    <img src=".github/hrms-attendance.png"/>
    <img src=".github/hrms-salary.png"/>
    <img src=".github/hrms-pwa.png"/>
</details>

## Technology Under the Hood

Frappe HR is built upon powerful, open-source technologies:

*   **Frappe Framework:** ( [Frappe Framework](https://github.com/frappe/frappe) ) A full-stack web application framework written in Python and Javascript, providing a solid foundation for building feature-rich HR applications.
*   **Frappe UI:** ( [Frappe UI](https://github.com/frappe/frappe-ui) ) A Vue-based UI library, offering a modern and responsive user interface for a seamless user experience.

## Production Setup

### Managed Hosting - Frappe Cloud

Simplify your deployment with [Frappe Cloud](https://frappecloud.com). It's a user-friendly platform for hosting Frappe applications, taking care of installation, upgrades, and maintenance.

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

Follow these steps to set up Frappe HR using Docker:

1.  Ensure you have Docker, docker-compose, and git installed.
2.  Clone the repository: `git clone https://github.com/frappe/hrms`
3.  Navigate to the Docker directory: `cd hrms/docker`
4.  Run `docker-compose up`
5.  Once the setup script completes, access Frappe HR at `http://localhost:8000`
6.  Login with:
    *   Username: `Administrator`
    *   Password: `admin`

### Local Setup

Follow these steps to set up Frappe HR locally:

1.  Set up Bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server.
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

## Learning and Community

*   [Frappe School](https://frappe.school) - Learn about the Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Explore comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get real-time support.

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
Key improvements and explanations:

*   **SEO-Optimized Title and Hook:**  The title now directly includes key search terms like "Open-Source HR" and "HRMS Software." The introductory sentence (hook) is also more compelling and descriptive.  It uses a strong action word (Manage).
*   **Clear Headings:**  Uses `##` for main section headings, enhancing readability and SEO.
*   **Bulleted Key Features:**  Uses bullet points for easy scanning and readability, emphasizing the core functionalities.
*   **Concise Language:** The content is streamlined and avoids unnecessary phrasing.
*   **Keywords:** Includes relevant keywords throughout the text (e.g., "HRMS," "Human Resource Management System," "open-source," "payroll").
*   **Context and Value Proposition:** The "About" section provides the "why" behind the software, adding value and addressing the user's needs.
*   **Call to Action (Implicit):** The descriptions of the features implicitly encourage the user to explore the software further.
*   **Community and Learning:** The "Learning and Community" section gives the user immediate links to resources, demonstrating the software's support network.
*   **Clear Instructions:**  Setup instructions are concise.
*   **Links Back to Repo:**  Included a link back to the original repo in the headings.
*   **Maintain structure:** Maintained the original structure for consistency.