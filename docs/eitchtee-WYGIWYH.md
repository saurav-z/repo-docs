<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Take Control of Your Finances with a Simple, Powerful Tracker
  <br>
</h1>

<h4 align="center">An open-source, principles-first finance tracker for straightforward money management.</h4>

<p align="center">
  <a href="#why-wygiwyh">Why WYGIWYH?</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#how-to-use">How to Use</a> •
  <a href="#how-it-works">How it Works</a> •
  <a href="#contributing">Contribute</a> •
  <a href="#caveats-and-warnings">Caveats and Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**WYGIWYH** (What You Get Is What You Have) is a finance tracker built on a no-budget, principles-first approach, empowering you to manage your money with simplicity and flexibility. Forget complex budgeting; WYGIWYH helps you track income, expenses, and investments with ease.

<img src=".github/img/monthly_view.png" width="18%"></img> <img src=".github/img/yearly.png" width="18%"></img> <img src=".github/img/networth.png" width="18%"></img> <img src=".github/img/calendar.png" width="18%"></img> <img src=".github/img/all_transactions.png" width="18%"></img>

## Why WYGIWYH?

Tired of overly complex budgeting apps? WYGIWYH simplifies finance tracking with a straightforward principle:

> Use what you earn this month for this month. Savings are tracked but treated as untouchable for future months.

This approach allows you to avoid dipping into savings while still getting a clear view of your spending. WYGIWYH was created to address the shortcomings of existing financial tools by offering:

*   Multi-currency support
*   No rigid budgeting constraints
*   Web app usability with mobile-friendly design
*   API for automation and integration
*   Custom transaction rules

## Key Features

**WYGIWYH** simplifies your personal finance tracking with the following:

*   **Unified Transaction Tracking:** Organize all income and expenses in one place.
*   **Multi-Account Support:** Track money and assets across various accounts (banks, wallets, investments, etc.).
*   **Multi-Currency Support:** Manage transactions and balances in different currencies dynamically.
*   **Custom Currencies:** Create your own currencies for crypto, rewards points, or other models.
*   **Automated Adjustments with Rules:** Automatically modify transactions based on customizable rules.
*   **Built-in Dollar-Cost Average (DCA) Tracker:** Ideal for tracking recurring investments.
*   **API Support:** Seamlessly integrate with external services for automation and data synchronization.

## Demo

Try out WYGIWYH with the demo credentials:

> [!NOTE]
> **E-mail:** `demo@demo.com`
> **Password:** `wygiwyhdemo`

**Important:** Any data added to the demo will be wiped within 24 hours. Many automation features are disabled in the demo.

[Try the WYGIWYH Demo](https://wygiwyh-demo.herculino.com/)

## How to Use

WYGIWYH can be run using Docker and docker-compose. Follow the instructions below to set up your instance:

```bash
# Create a folder for WYGIWYH (optional)
$ mkdir WYGIWYH

# Go into the folder
$ cd WYGIWYH

$ touch docker-compose.yml
$ nano docker-compose.yml
# Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml and edit according to your needs

# Fill the .env file with your configurations
$ touch .env
$ nano .env # or any other editor you want to use
# Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/.env.example and edit accordingly

# Run the app
$ docker compose up -d

# Create the first admin account. This isn't required if you set the enviroment variables: ADMIN_EMAIL and ADMIN_PASSWORD.
$ docker compose exec -it web python manage.py createsuperuser
```

### Running Locally

To run WYGIWYH locally, adjust the following in your `.env` file:

1.  Remove `URL`
2.  Set `HTTPS_ENABLED` to `false`
3.  Keep the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 \[::1])

You can then access WYGIWYH at `localhost:OUTBOUND_PORT`.

> [!NOTE]
>
> *   If running behind Tailscale or similar, add your machine's IP to `DJANGO_ALLOWED_HOSTS`.
> *   For IPs other than localhost, add them to `DJANGO_ALLOWED_HOSTS`, without `http://`.

### Latest Changes

Features are added to the `main` branch when ready. For the latest version, build from source or use the `:nightly` tag with Docker. Be aware of potential undocumented breaking changes.

All Dockerfiles are located [here](https://github.com/eitchtee/WYGIWYH/tree/main/docker/prod).

### Unraid

[nwithan8](https://github.com/nwithan8) provides a Unraid template for WYGIWYH. Explore it in the [unraid_templates](https://github.com/nwithan8/unraid_templates) repo.

WYGIWYH is also available on the Unraid Store. You must provision your own PostgreSQL database (version 15 or up).

To create your first user on Unraid, open the container's console, then run `python manage.py createsuperuser`.

## Environment Variables

Configure WYGIWYH using these environment variables:

| Variable                      | Type        | Default                           | Description                                                                                                                                                                                              |
|-------------------------------|-------------|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DJANGO_ALLOWED_HOSTS`          | string      | localhost 127.0.0.1               | A list of space separated domains and IPs representing the host/domain names that WYGIWYH site can serve. [Click here](https://docs.djangoproject.com/en/5.1/ref/settings/#allowed-hosts) for more details                               |
| `HTTPS_ENABLED`                 | true\|false | false                             | Whether to use secure cookies. If this is set to true, the cookie will be marked as “secure”, which means browsers may ensure that the cookie is only sent under an HTTPS connection                                                     |
| `URL`                           | string      | http://localhost http://127.0.0.1 | A list of space separated domains and IPs (with the protocol) representing the trusted origins for unsafe requests (e.g. POST). [Click here](https://docs.djangoproject.com/en/5.1/ref/settings/#csrf-trusted-origins ) for more details |
| `SECRET_KEY`                    | string      | ""                                | This is used to provide cryptographic signing, and should be set to a unique, unpredictable value.                                                                                                                                       |
| `DEBUG`                         | true\|false | false                             | Turns DEBUG mode on or off, this is useful to gather more data about possible errors you're having. Don't use in production.                                                                                                             |
| `SQL_DATABASE`                  | string      | None *required                    | The name of your postgres database                                                                                                                                                                                                       |
| `SQL_USER`                      | string      | user                              | The username used to connect to your postgres database                                                                                                                                                                                   |
| `SQL_PASSWORD`                  | string      | password                          | The password used to connect to your postgres database                                                                                                                                                                                   |
| `SQL_HOST`                      | string      | localhost                         | The address used to connect to your postgres database                                                                                                                                                                                    |
| `SQL_PORT`                      | string      | 5432                              | The port used to connect to your postgres database                                                                                                                                                                                       |
| `SESSION_EXPIRY_TIME`           | int         | 2678400 (31 days)                 | The age of session cookies, in seconds. E.g. how long you will stay logged in                                                                                                                                                            |
| `ENABLE_SOFT_DELETE`            | true\|false | false                             | Whether to enable transactions soft delete, if enabled, deleted transactions will remain in the database. Useful for imports and avoiding duplicate entries.                                                                             |
| `KEEP_DELETED_TRANSACTIONS_FOR` | int         | 365                               | Time in days to keep soft deleted transactions for. If 0, will keep all transactions indefinitely. Only works if ENABLE_SOFT_DELETE is true.                                                                                             |
| `TASK_WORKERS`                  | int         | 1                                 | How many workers to have for async tasks. One should be enough for most use cases                                                                                                                                                        |
| `DEMO`                          | true\|false | false                             | If demo mode is enabled.                                                                                                                                                                                                                 |
| `ADMIN_EMAIL`                   | string      | None                              | Automatically creates an admin account with this email. Must have `ADMIN_PASSWORD` also set.                                                                                                                                             |
| `ADMIN_PASSWORD`                | string      | None                              | Automatically creates an admin account with this password. Must have `ADMIN_EMAIL` also set.                                                                                                                                             |
| `CHECK_FOR_UPDATES`             | bool        | true                              | Check and notify users about new versions. The check is done by doing a single query to Github's API every 12 hours.                                                                                  |

## OIDC Configuration

Configure OpenID Connect (OIDC) login through `django-allauth`:

| Variable             | Description                                                                                                                                                                                                                                            |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OIDC_CLIENT_NAME`   | The name of the provider. will be displayed in the login page. Defaults to `OpenID Connect`                                                                                                                                                            |
| `OIDC_CLIENT_ID`     | The Client ID provided by your OIDC provider.                                                                                                                                                                                                          |
| `OIDC_CLIENT_SECRET` | The Client Secret provided by your OIDC provider.                                                                                                                                                                                                      |
| `OIDC_SERVER_URL`    | The base URL of your OIDC provider's discovery document or authorization server (e.g., `https://your-provider.com/auth/realms/your-realm`). `django-allauth` will use this to discover the necessary endpoints (authorization, token, userinfo, etc.). |
| `OIDC_ALLOW_SIGNUP`  | Allow the automatic creation of inexistent accounts on a successfull authentication. Defaults to `true`.                                                                                                                                               |

**Callback URL (Redirect URI):**

When configuring your OIDC provider, you will need to provide a callback URL (also known as a Redirect URI). For WYGIWYH, the default callback URL is:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with the actual URL where your WYGIWYH instance is accessible. And `<OIDC_CLIENT_NAME>` with the slugfied value set in OIDC_CLIENT_NAME or the default `openid-connect` if you haven't set this variable.

## How it Works

For more in-depth information, explore the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Contributing

Help translate WYGIWYH!

<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

> [!NOTE]
> Login with your github account

## Caveats and Warnings

*   I'm not an accountant, and some terms or calculations might be inaccurate. Please open an issue if you find any areas for improvement.
*   Most calculations are performed at runtime, which could impact performance. On my personal instance with 3000+ transactions and 4000+ exchange rates, page load times average about 500ms.
*   WYGIWYH isn't a budgeting or double-entry accounting application. If you require those features, explore other options. If you really need them in WYGIWYH, start a discussion.

## Built With

WYGIWYH is built using these open-source tools:

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