[![Django Ledger Logo](https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png)](https://github.com/arrobalytics/django-ledger)

# Django Ledger: Powerful Double-Entry Accounting for Django

**Django Ledger** is a robust and flexible financial management system built specifically for the Django web framework, offering a comprehensive solution for your accounting needs.

[Get Started Guide](https://www.djangoledger.com/get-started) | [Join our Discord](https://discord.gg/c7PZcbYgrc) | [Documentation](https://django-ledger.readthedocs.io/en/latest/) | [QuickStart Notebook](https://github.com/arrobalytics/django-ledger/blob/develop/notebooks/QuickStart%20Notebook.ipynb)

## Key Features:

*   **Double-Entry Accounting:** Ensures accuracy and financial integrity.
*   **Hierarchical Chart of Accounts:** Organize your financial data effectively.
*   **Financial Statements:** Generate Income Statements, Balance Sheets, and Cash Flow statements.
*   **Transaction Management:** Includes Ledgers, Journal Entries, and Transactions.
*   **Comprehensive Financial Document Support:** Includes Purchase Orders, Sales Orders, Bills, and Invoices.
*   **Financial Ratio Calculations:** Analyze your financial performance with ease.
*   **Multi-tenancy Support:**  Supports managing multiple businesses or entities.
*   **Data Import:** OFX & QFX file import
*   **Inventory Management:** Track and manage your inventory.
*   **Units of Measure:** Consistent handling of various units.
*   **Bank Account Integration:** Store and access bank account details.
*   **Django Admin Integration:** Seamlessly integrates with Django's admin panel.
*   **Built-in Entity Management UI:** Manage and view your entities via a user-friendly interface.

## Getting Involved

We welcome contributions from the community!  Help us improve Django Ledger.

*   **Feature Requests/Bug Reports:** Open an issue in the repository.
*   **Customization/Consulting:** [Contact us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com
*   **Contribute:** See our [contribution guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md)

## Who Should Contribute?

We are looking for contributors with:

*   Python and Django programming skills
*   Finance and accounting expertise
*   Interest in developing a robust accounting engine API

If you have experience in these areas, your contributions are highly valued.

## Installation

Django Ledger is a Django application.  You need a working Django project and familiarity with Django to use Django Ledger.

The easiest way to get started is with the zero-config [Django Ledger starter template](https://github.com/arrobalytics/django-ledger-starter).  Alternatively, you can install it in an existing project:

1.  **Add to `INSTALLED_APPS`:**

    ```python
    INSTALLED_APPS = [
        ...,
        'django_ledger',
        ...,
    ]
    ```

2.  **Add Context Preprocessor:**

    ```python
    TEMPLATES = [
        {
            'OPTIONS': {
                'context_processors': [
                    '...',
                    'django_ledger.context.django_ledger_context'  # Add this line
                ],
            },
        },
    ]
    ```

3.  **Run Migrations:**

    ```bash
    python manage.py migrate
    ```

4.  **Add URLs to `urls.py`:**

    ```python
    from django.urls import include, path

    urlpatterns = [
        ...,
        path('ledger/', include('django_ledger.urls', namespace='django_ledger')),
        ...,
    ]
    ```

5.  **Run your project:**

    ```bash
    python manage.py runserver
    ```

*   Navigate to the Django Ledger root view (usually http://127.0.0.1:8000/ledger).
*   Use your superuser credentials to log in.

## Deprecated Behavior Setting (v0.8.0+)

The `DJANGO_LEDGER_USE_DEPRECATED_BEHAVIOR` setting controls access to deprecated features.

*   **Default:** `False` (deprecated features are disabled)
*   Set to `True` in your Django settings to temporarily use deprecated features.

## Setting Up Django Ledger for Development

Django Ledger comes with a development environment under the `dev_env/` folder.

1.  Clone the repository:

    ```bash
    git clone https://github.com/arrobalytics/django-ledger.git && cd django-ledger
    ```

2.  Install PipEnv (if not already installed):

    ```bash
    pip install -U pipenv
    ```

3.  Create and activate the virtual environment:

    ```bash
    pipenv install
    pipenv shell
    ```

4.  Apply migrations:

    ```bash
    python manage.py migrate
    ```

5.  Create a superuser:

    ```bash
    python manage.py createsuperuser
    ```

6.  Run the development server:

    ```bash
    python manage.py runserver
    ```

## Setting Up Django Ledger for Development using Docker

1.  Give executable permissions to `entrypoint.sh`:

    ```bash
    sudo chmod +x entrypoint.sh
    ```

2.  Add `0.0.0.0` to `ALLOWED_HOSTS` in `settings.py`.

3.  Build and run the container:

    ```bash
    docker compose up --build
    ```

4.  Create a superuser (in a separate terminal):

    ```bash
    docker ps
    docker exec -it <container_id> /bin/sh
    python manage.py createsuperuser
    ```

5.  Access the application at http://0.0.0.0:8000/.

## Run Test Suite

After setting up your development environment:

```bash
python manage.py test django_ledger
```

## Screenshots

**(Image links remain the same)**
![django ledger entity dashboard](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_entity_dashboard.png)
![django ledger balance sheet](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_income_statement.png)
![django ledger income statement](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_balance_sheet.png)
![django ledger bill](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_bill.png)
![django ledger invoice](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_invoice.png)

## Financial Statements Screenshots

**(Image links remain the same)**
![balance_sheet_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/BalanceSheetStatement.png)
![income_statement_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/IncomeStatement.png)
![cash_flow_statement_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/CashFlowStatement.png)
```

Key improvements:

*   **SEO Optimization:**  Includes relevant keywords ("Django", "accounting", "financial management") in headings and text.
*   **Clear Structure:** Uses headings and subheadings for readability.
*   **Concise Language:** Streamlines the original text.
*   **Benefit-driven:**  Focuses on *what* the features *do* for the user.
*   **Call to Action:**  Encourages community involvement.
*   **Complete Installation Instructions:** Better flow and formatting for quick setup.
*   **Docker Instructions:** Improved Docker setup.
*   **Clearer Formatting:** Use of bolding and bullet points for emphasis.
*   **Link Back:** Includes a prominent link back to the original repository.
*   **One-Sentence Hook:** Uses the opening sentence to get the user's attention.
*   **Uses H2's instead of H1's:** So as not to override the title