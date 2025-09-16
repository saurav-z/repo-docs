![Django Ledger Logo](https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png)

# Django Ledger: Powerful Double-Entry Accounting for Django

**Django Ledger** empowers developers with a robust, easy-to-use accounting engine directly within the Django framework.  For more information, please visit the original repository: [https://github.com/arrobalytics/django-ledger](https://github.com/arrobalytics/django-ledger).

[FREE Get Started Guide](https://www.djangoledger.com/get-started) | [Join our Discord](https://discord.gg/c7PZcbYgrc) | [Documentation](https://django-ledger.readthedocs.io/en/latest/) | [QuickStart Notebook](https://github.com/arrobalytics/django-ledger/blob/develop/notebooks/QuickStart%20Notebook.ipynb)

## Key Features

*   **Double-Entry Accounting:** Ensures accuracy and transparency in financial tracking.
*   **Hierarchical Chart of Accounts:** Organize your finances with flexible account structures.
*   **Financial Reporting:** Generate essential statements like Income Statements, Balance Sheets, and Cash Flow statements.
*   **Transaction Management:**  Supports Ledgers, Journal Entries, and Transactions for detailed record-keeping.
*   **Order and Invoice Handling:**  Manage financial transactions with Purchase Orders, Sales Orders, Bills, and Invoices.
*   **Financial Analysis:** Includes built-in financial ratio calculations.
*   **Multi-Tenancy Support:**  Allows for managing multiple entities or organizations.
*   **Data Import:** Integrates with OFX & QFX file formats.
*   **Inventory and Unit of Measure Management:** Keep track of products and inventory.
*   **Django Admin Integration:** Seamlessly integrates with the Django Admin interface for easy data management.
*   **Built-in Entity Management UI:** Manage organizations and tenants.

## Getting Involved

We welcome contributions of all kinds!

*   **Feature Requests/Bug Reports:**  Open an issue in the repository.
*   **Customization/Consulting:**  [Contact us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com.
*   **Contribute:** See our [contribution guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md).

## Who Should Contribute?

We encourage contributions from individuals with:

*   Python and Django programming skills.
*   Finance and accounting expertise.
*   A passion for building a robust accounting engine.

## Installation

Django Ledger is a Django application.  Ensure you have a working Django project before installing.

The easiest way to start is to use the zero-config Django Ledger starter template, found [here](https://github.com/arrobalytics/django-ledger-starter).

### Adding Django Ledger to an Existing Project

1.  **Add `django_ledger` to `INSTALLED_APPS`:**

    ```python
    INSTALLED_APPS = [
        ...,
        'django_ledger',
        ...,
    ]
    ```

2.  **Add Django Ledger Context Preprocessor:**

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

3.  **Perform Database Migrations:**

    ```bash
    python manage.py migrate
    ```

4.  **Add URLs to your project's `urls.py`:**

    ```python
    from django.urls import include, path

    urlpatterns = [
        ...,
        path('ledger/', include('django_ledger.urls', namespace='django_ledger')),
        ...,
    ]
    ```

5.  **Run Your Project:**

    ```bash
    python manage.py runserver
    ```

6.  **Access Django Ledger:** Navigate to the root view in your project's `urlpatterns` setting (e.g., http://127.0.0.1:8000/ledger) and log in using your superuser credentials.

## Deprecated Behavior Setting (v0.8.0+)

Version 0.8.0 introduces the `DJANGO_LEDGER_USE_DEPRECATED_BEHAVIOR` setting to control deprecated features.  Set to `True` in your Django settings to temporarily use deprecated features during transition. Defaults to `False`.

## Setting Up Django Ledger for Development

Django Ledger provides a development environment in the `dev_env/` folder (not for production).

1.  Navigate to your project directory.
2.  Clone the repository:

    ```bash
    git clone https://github.com/arrobalytics/django-ledger.git && cd django-ledger
    ```
3. Install PipEnv, if not already installed:

    ```bash
    pip install -U pipenv
    ```

4. Create virtual environment.

    ```bash
    pipenv install
    ```

    Or specify a Python version:

    ```bash
    pipenv install --python PATH_TO_INTERPRETER
    ```

5. Activate environment.

    ```bash
    pipenv shell
    ```

6. Apply migrations.

    ```bash
    python manage.py migrate
    ```

7.  Create a superuser:

    ```bash
    python manage.py createsuperuser
    ```

8.  Run the development server:

    ```bash
    python manage.py runserver
    ```

## How To Set Up Django Ledger for Development using Docker

1.  Navigate to your project directory.

2.  Give executable permissions to entrypoint.sh

    ```bash
    sudo chmod +x entrypoint.sh
    ```

3.  Add host '0.0.0.0' into ALLOWED_HOSTS in settings.py.

4.  Build the image and run the container.

    ```bash
    docker compose up --build
    ```

5.  Add Django Superuser by running command in seprate terminal

    ```bash
    docker ps
    ```

    Select container id of running container and execute following command

    ```bash
    docker exec -it containerId /bin/sh
    ```

    ```bash
    python manage.py createsuperuser
    ```

6.  Navigate to http://0.0.0.0:8000/ on browser.

## Run Test Suite

After setting up your development environment you may run tests.

```bash
python manage.py test django_ledger
```

## Screenshots

*   ![Django Ledger Entity Dashboard](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_entity_dashboard.png)
*   ![Django Ledger Balance Sheet](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_income_statement.png)
*   ![Django Ledger Income Statement](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_balance_sheet.png)
*   ![Django Ledger Bill](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_bill.png)
*   ![Django Ledger Invoice](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_invoice.png)

## Financial Statements Screenshots

*   ![Balance Sheet Report](https://django-ledger.us-east-1.linodeobjects.com/public/img/BalanceSheetStatement.png)
*   ![Income Statement Report](https://django-ledger.us-east-1.linodeobjects.com/public/img/IncomeStatement.png)
*   ![Cash Flow Statement Report](https://django-ledger.us-east-1.linodeobjects.com/public/img/CashFlowStatement.png)