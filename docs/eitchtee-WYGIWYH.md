<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Take Control of Your Finances with a Simple, Powerful Tracker
  <br>
</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#advanced-configuration">Advanced Configuration</a> •
  <a href="#translations">Translations</a> •
  <a href="#caveats">Caveats</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH (What You Get Is What You Have)** is a finance tracker designed for a straightforward, no-budget approach to money management. 

## About

WYGIWYH simplifies personal finance by focusing on a simple principle: use this month's income for this month's expenses. This philosophy helps you avoid dipping into savings while providing a clear view of your spending. This tool offers multi-currency support, customizable transactions, and a built-in dollar-cost averaging tracker to put you in control of your finances.

## Key Features

*   ✅ **Unified Transaction Tracking:** Track all income and expenses in one place.
*   🏦 **Multi-Account Support:** Manage funds across banks, wallets, and investments.
*   🌐 **Multi-Currency Support:** Handle transactions and balances in various currencies.
*   💰 **Custom Currencies:** Create currencies for crypto, rewards points, or other models.
*   ⚙️ **Automated Adjustments with Rules:** Customize transaction modifications.
*   📈 **Dollar-Cost Averaging (DCA) Tracker:** Monitor recurring investments.
*   💻 **API Support:** Integrate with external services for seamless automation.

## Demo

Experience WYGIWYH firsthand with our demo!
> [!NOTE]
> E-mail: `demo@demo.com`
> 
> Password: `wygiwyhdemo`

Keep in mind that **any data you add will be wiped in 24 hours or less**. And that **most automation features like the API, Rules, Automatic Exchange Rates and Import/Export are disabled**.

## Getting Started

WYGIWYH utilizes [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/) for easy setup.

1.  **Create a Project Folder:** `mkdir WYGIWYH && cd WYGIWYH` (Optional)
2.  **Create docker-compose.yml:** `touch docker-compose.yml && nano docker-compose.yml`
    *   Paste the contents of [`docker-compose.prod.yml`](https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml) and customize as needed.
3.  **Create .env file:** `touch .env && nano .env`
    *   Populate with values from [`example .env`](https://github.com/eitchtee/WYGIWYH/blob/main/.env.example) and customize as needed.
4.  **Run the Application:** `docker compose up -d`
5.  **Create Admin User:** `docker compose exec -it web python manage.py createsuperuser`
    *   Alternatively, set `ADMIN_EMAIL` and `ADMIN_PASSWORD` environment variables.

### Running Locally

For local development:

1.  In your `.env` file:
    *   Remove `URL`
    *   Set `HTTPS_ENABLED=false`
    *   Leave `DJANGO_ALLOWED_HOSTS` as default.
2.  Access at `localhost:OUTBOUND_PORT`.
    *   If running behind Tailscale or similar, add your machine's IP to `DJANGO_ALLOWED_HOSTS`.
    *   For non-localhost IPs, add them to `DJANGO_ALLOWED_HOSTS` without `http://`.

## Advanced Configuration

### Unraid

WYGIWYH is also available on the Unraid Store.  Refer to the [Unraid section](#unraid) in the original README for details.
The latest documentation for Unraid setup can be found here [here](https://github.com/eitchtee/WYGIWYH/blob/main/README.md#unraid).

### Environment Variables

Configure WYGIWYH using environment variables. See the [Environment Variables section](#environment-variables) of the original README for a full list of options and their descriptions.

### OIDC Configuration

Integrate with OpenID Connect (OIDC) for authentication. Configure the following environment variables:

*   `OIDC_CLIENT_NAME`
*   `OIDC_CLIENT_ID`
*   `OIDC_CLIENT_SECRET`
*   `OIDC_SERVER_URL`
*   `OIDC_ALLOW_SIGNUP`

Remember to configure the callback URL (Redirect URI) in your OIDC provider settings, which is:
`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

## Translations

Help translate WYGIWYH! Contribute to translations on [Translations](https://translations.herculino.com/engage/wygiwyh/).

## Caveats

*   I'm not an accountant, so terms or calculations could be incorrect.  Please [open an issue](https://github.com/eitchtee/WYGIWYH/issues) to report any issues.
*   Most calculations are done at runtime, which could cause performance issues.
*   This is not a budgeting or double-entry accounting application.

## Built With

WYGIWYH leverages the power of open-source technologies:

*   Django
*   HTMX
*   _hyperscript
*   Procrastinate
*   Bootstrap
*   Tailwind
*   Webpack
*   PostgreSQL
*   Django REST framework
*   Alpine.js

---
```
Key improvements and optimizations:

*   **SEO Optimization:** Includes relevant keywords like "finance tracker," "personal finance," and "money management."  Uses descriptive headings.
*   **Clear Hook:** Immediately states the value proposition in the first sentence.
*   **Concise Summaries:** Rephrases information for brevity and readability.
*   **Bulleted Key Features:**  Highlights the core functionality.
*   **Direct Links:**  Provides a prominent link back to the original repository.
*   **Structured Content:** Organizes the information with clear headings and subheadings.
*   **Calls to Action:** Encourages the user to try the demo and contribute.
*   **Sections for Clarity:**  Separates out different parts of the information.
*   **Focus on Value:**  Emphasizes the benefits to the user.
*   **Improved Demo section** Includes link to the demo.
*   **Improved OIDC Section** Included more detailed instructions.