# WYGIWYH: Take Control of Your Finances with a Simple and Flexible Tracker

> Simplify your finances with WYGIWYH, a powerful, open-source finance tracker built on a no-budget, principles-first approach. [Check it out on GitHub!](https://github.com/eitchtee/WYGIWYH)

<img align="center" src="./.github/img/logo.png" alt="WYGIWYH Logo" width="200">

**WYGIWYH** (_What You Get Is What You Have_) is a powerful and opinionated finance tracker, designed for those who prefer a straightforward approach to managing their money. This tool empowers you to track your income and expenses across multiple currencies, accounts, and investments.  

**Key Features:**

*   ✅ **Unified Transaction Tracking:**  Keep all income and expenses organized in one place.
*   ✅ **Multi-Account Support:** Track finances across banks, wallets, and investment accounts.
*   ✅ **Multi-Currency Support:**  Seamlessly manage transactions and balances in different currencies.
*   ✅ **Customizable Currencies:** Create and track custom currencies like crypto or reward points.
*   ✅ **Automated Transaction Rules:**  Apply rules for automatic adjustments (e.g., credit card billing).
*   ✅ **Built-in Dollar-Cost Averaging (DCA) Tracker:** Track recurring investments effectively.
*   ✅ **API Support:** Integrate with external services for automated transaction synchronization.

## Why Choose WYGIWYH?

WYGIWYH is designed for simplicity. It operates on a core principle:

> Use what you earn this month for this month. Savings are tracked but treated as untouchable for future months.

This approach eliminates the complexities of budgeting while still allowing you to monitor where your money goes, making financial management more accessible and less overwhelming.

## Demo

Experience WYGIWYH firsthand: [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

*   **Email:** `demo@demo.com`
*   **Password:** `wygiwyhdemo`

**Note:** Demo data is reset daily.  Automation features are disabled in the demo.

## How to Get Started

WYGIWYH uses Docker and docker-compose.

**Steps:**

1.  **Install Docker:**  Ensure Docker and docker-compose are installed on your system. ([Docker Installation Guide](https://docs.docker.com/engine/install/))
2.  **Create a `docker-compose.yml` File:**  Use the `docker-compose.prod.yml` file as a base and modify it according to your needs.
3.  **Configure Your `.env` File:**  Customize environment variables based on your needs using the `.env.example` file as a template.
4.  **Run the Application:** Execute `docker compose up -d` from the command line.
5.  **Create Admin User:** Run `docker compose exec -it web python manage.py createsuperuser` to create your first admin account, if you haven't set the `ADMIN_EMAIL` and `ADMIN_PASSWORD` environment variables.

**[See the full usage documentation for detailed instructions.](https://github.com/eitchtee/WYGIWYH#how-to-use)**

## Advanced Configuration

### Environment Variables

Customize WYGIWYH's behavior with these environment variables:

*   **DJANGO\_ALLOWED\_HOSTS:** Define allowed domains/IPs.
*   **HTTPS\_ENABLED:** Enables secure cookies.
*   **URL:** Define trusted origins for requests.
*   **SECRET\_KEY:**  Set a unique secret key for security.
*   **DEBUG:** Toggle debug mode on or off.
*   **SQL\_DATABASE, SQL\_USER, SQL\_PASSWORD, SQL\_HOST, SQL\_PORT:** Database connection settings.
*   **SESSION\_EXPIRY\_TIME:** Session duration in seconds.
*   **ENABLE\_SOFT\_DELETE:** Enables soft deletion for transactions.
*   **KEEP\_DELETED\_TRANSACTIONS\_FOR:** Time to keep soft-deleted transactions.
*   **TASK\_WORKERS:** Number of workers for async tasks.
*   **DEMO:** Enable or disable demo mode.
*   **ADMIN\_EMAIL, ADMIN\_PASSWORD:** Create an admin user on startup.
*   **CHECK\_FOR\_UPDATES:** Enable/disable version update notifications.

### OIDC Configuration

Integrate with OpenID Connect (OIDC) providers for user authentication.  Configure these environment variables:

*   `OIDC_CLIENT_NAME`: Name of the provider. Defaults to `OpenID Connect`.
*   `OIDC_CLIENT_ID`: The Client ID from your provider.
*   `OIDC_CLIENT_SECRET`: The Client Secret from your provider.
*   `OIDC_SERVER_URL`: The base URL of your OIDC provider.
*   `OIDC_ALLOW_SIGNUP`: Allow for signup. Defaults to `true`.

**Callback URL (Redirect URI):**
`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/` (replace `<OIDC_CLIENT_NAME>` with the slugfied version of `OIDC_CLIENT_NAME` or the default value)

## Contribution & Translation

Help translate WYGIWYH into your language:
<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

## Important Considerations

*   **Not for Accountants:**  This tool is designed with a principles-first approach.
*   **Performance:** Calculations are done at runtime, potentially impacting performance.
*   **Feature Requests:**  Open a discussion if you need budgeting or double-entry accounting.

## Built With

WYGIWYH is built with a range of powerful open-source tools, including:

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