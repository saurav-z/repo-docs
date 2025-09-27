<!-- django ledger logo -->
![django ledger logo](https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png)

# Django Ledger: Powerful Accounting for Django

**Django Ledger** is a robust, open-source accounting engine that seamlessly integrates with your Django applications, simplifying complex financial tasks. [Explore the Django Ledger repository](https://github.com/arrobalytics/django-ledger).

## Key Features:

*   **Double-Entry Accounting:** Ensures accurate financial record-keeping.
*   **Hierarchical Chart of Accounts:** Organize your financial data effectively.
*   **Financial Statements:** Generate Income Statements, Balance Sheets, and Cash Flow statements.
*   **Order Management:** Includes Purchase Orders, Sales Orders, Bills, and Invoices.
*   **Financial Ratio Calculations:** Analyze key financial metrics.
*   **Multi-tenancy Support:** Manage multiple organizations or entities.
*   **Ledgers, Journal Entries & Transactions:** Track every financial movement.
*   **OFX & QFX File Import:** Easily import financial data.
*   **Inventory Management:** Track and manage stock levels.
*   **Unit of Measures:** Define and manage units for inventory and more.
*   **Bank Account Information:** Seamless integration of bank data.
*   **Django Admin Integration:** Easily manage and interact with data through the Django admin interface.
*   **Built-in Entity Management UI:** Manage your entities easily.

## Getting Started

To use Django Ledger, you'll need:

*   A working knowledge of Django and a Django project.
*   Refer to the Django version you are using.
*   The easiest way to start is to use the zero-config Django Ledger starter template. See details [here](https://github.com/arrobalytics/django-ledger-starter).

### Installation

1.  **Add `django_ledger` to `INSTALLED_APPS` in your `settings.py`:**

    ```python
    INSTALLED_APPS = [
        ...,
        'django_ledger',
        ...,
    ]
    ```

2.  **Add Django Ledger Context Preprocessor in `settings.py`:**

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

3.  **Run database migrations:**

    ```bash
    python manage.py migrate
    ```

4.  **Include Django Ledger URLs in your project's `urls.py`:**

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

6.  **Access Django Ledger:** Navigate to the URL assigned in your project's `urlpatterns` (typically `http://127.0.0.1:8000/ledger`). Log in using your superuser credentials.

## Deprecated Behavior (v0.8.0+)

The `DJANGO_LEDGER_USE_DEPRECATED_BEHAVIOR` setting controls access to deprecated features. By default, these features are disabled. To temporarily keep using them, set this to `True` in your Django settings.

## Development Setup

**To contribute, follow these steps:**

1.  Clone the repository:

    ```bash
    git clone https://github.com/arrobalytics/django-ledger.git && cd django-ledger
    ```

2.  Install PipEnv:

    ```bash
    pip install -U pipenv
    ```

3.  Create and activate a virtual environment:

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

## Development Setup with Docker

1.  Navigate to your project directory.

2.  Give executable permissions to `entrypoint.sh`:

    ```bash
    sudo chmod +x entrypoint.sh
    ```

3.  Add host '0.0.0.0' into `ALLOWED_HOSTS` in `settings.py`.

4.  Build and run the Docker container:

    ```bash
    docker compose up --build
    ```

5.  Create a Django superuser:

    ```bash
    docker ps
    docker exec -it <containerId> /bin/sh
    python manage.py createsuperuser
    ```

6.  Access Django Ledger at `http://0.0.0.0:8000/`.

## Running Tests

```bash
python manage.py test django_ledger
```

## Screenshots

*   ![django ledger entity dashboard](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_entity_dashboard.png)
*   ![django ledger balance sheet](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_income_statement.png)
*   ![django ledger income statement](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_balance_sheet.png)
*   ![django ledger bill](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_bill.png)
*   ![django ledger invoice](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_invoice.png)

## Financial Statements Screenshots

*   ![balance_sheet_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/BalanceSheetStatement.png)
*   ![income_statement_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/IncomeStatement.png)
*   ![cash_flow_statement_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/CashFlowStatement.png)

## Contributing

We welcome contributions! Please review our [contribution guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md) for details.

*   **Feature Requests/Bug Reports:** Open an issue in the repository.
*   **For customization, advanced features, and consulting:** [Contact us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com

## Who Should Contribute?

We encourage contributions from individuals with:

*   Python and Django programming skills
*   Finance and accounting expertise
*   Interest in developing a robust accounting engine API