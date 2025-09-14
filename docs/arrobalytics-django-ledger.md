[![Django Ledger Logo](https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png)](https://github.com/arrobalytics/django-ledger)

# Django Ledger: Powerful Double-Entry Accounting for Django

**Django Ledger** provides a robust and user-friendly financial management system, simplifying complex accounting tasks within your Django applications.

[Get Started Guide](https://www.djangoledger.com/get-started) | [Join our Discord](https://discord.gg/c7PZcbYgrc) | [Documentation](https://django-ledger.readthedocs.io/en/latest/) | [QuickStart Notebook](https://github.com/arrobalytics/django-ledger/blob/develop/notebooks/QuickStart%20Notebook.ipynb)

## Key Features:

*   **Double-Entry Accounting:** Core accounting principles are implemented for accuracy.
*   **Hierarchical Chart of Accounts:** Organize your financial data effectively.
*   **Financial Statements:** Generate Income Statements, Balance Sheets, and Cash Flow statements.
*   **Transaction Management:** Supports Ledgers and Journal Entries.
*   **Order Management:** Includes Purchase Orders, Sales Orders, Bills, and Invoices.
*   **Import & Export:** Supports OFX & QFX file imports for easy data entry.
*   **Inventory Management:** Features Inventory management.
*   **Multi-Tenancy Support:** Manage multiple entities with ease.
*   **Django Admin Integration:** Seamlessly integrated into your Django admin panel.
*   **Financial Ratio Calculations:** Calculate key financial ratios for analysis.
*   **Built-in UI:** Includes a built-in Entity Management UI.
*   **Unit of Measures:** Handles Units of Measure.
*   **Bank Account Information:** Enables storing bank account information.
*   **Closing Entries:** Facilitates closing entry processes.

## Getting Involved & Contributing

We welcome contributions! Help us improve Django Ledger.

*   **Feature Requests/Bug Reports:** Open an issue in the [repository](https://github.com/arrobalytics/django-ledger).
*   **Customization & Consulting:** [Contact us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com for advanced features and services.
*   **Contribute:** Review our [contribution guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md) and submit pull requests.

## Who Should Contribute?

We are looking for contributors with:

*   Python and Django programming skills
*   Finance and accounting expertise
*   Interest in building a robust accounting engine API

If you have relevant experience, we welcome your pull requests.

## Installation

Django Ledger is a [Django](https://www.djangoproject.com/) application. Make sure you have a working Django project before you start.

The easiest way to start is to use the zero-config Django Ledger starter template. See details [here](https://github.com/arrobalytics/django-ledger-starter). Otherwise, you may create your project from scratch.

### Add Django Ledger to an Existing Project

1.  **Add to `INSTALLED_APPS`**:

    ```python
    INSTALLED_APPS = [
        ...,
        'django_ledger',
        ...,
    ]
    ```

2.  **Add Context Preprocessor**:

    ```python
    TEMPLATES = [
        {
            'OPTIONS': {
                'context_processors': [
                    '...',
                    'django_ledger.context.django_ledger_context'  # Add this line.
                ],
            },
        },
    ]
    ```

3.  **Run Migrations**:

    ```shell
    python manage.py migrate
    ```

4.  **Add URLs to your project's `urls.py`**:

    ```python
    from django.urls import include, path

    urlpatterns = [
        ...,
        path('ledger/', include('django_ledger.urls', namespace='django_ledger')),
        ...,
    ]
    ```

5.  **Run your project**:

    ```shell
    python manage.py runserver
    ```

    Navigate to the Django Ledger root view (typically `http://127.0.0.1:8000/ledger`). Use your superuser credentials to log in.

## Deprecated Behavior (v0.8.0+)

The `DJANGO_LEDGER_USE_DEPRECATED_BEHAVIOR` setting controls access to deprecated features. It defaults to `False`.  To enable these features temporarily, set the setting to `True`.

## Setting Up Django Ledger for Development

1.  Clone the repository and `cd` into the project directory.
    ```shell
    git clone https://github.com/arrobalytics/django-ledger.git && cd django-ledger
    ```
2.  Install PipEnv (if not already installed):
    ```shell
    pip install -U pipenv
    ```
3.  Create a virtual environment.
    ```shell
    pipenv install
    ```
    Or, if you have a specific python version
    ```shell
    pipenv install --python PATH_TO_INTERPRETER
    ```
4.  Activate the environment.
    ```shell
    pipenv shell
    ```
5.  Apply migrations.
    ```shell
    python manage.py migrate
    ```
6.  Create a development Django superuser.
    ```shell
    python manage.py createsuperuser
    ```
7.  Run the development server.
    ```shell
    python manage.py runserver
    ```

## How To Set Up Django Ledger for Development using Docker

1.  Navigate to your projects directory.
2.  Give executable permissions to entrypoint.sh
    ```shell
    sudo chmod +x entrypoint.sh
    ```
3.  Add host '0.0.0.0' into ALLOWED_HOSTS in settings.py.
4.  Build the image and run the container.
    ```shell
    docker compose up --build
    ```
5.  Add Django Superuser by running command in seprate terminal
    ```shell
    docker ps
    ```
    Select container id of running container and execute following command
    ```shell
    docker exec -it containerId /bin/sh
    ```
    ```shell
    python manage.py createsuperuser
    ```
6.  Navigate to http://0.0.0.0:8000/ on browser.

## Run Test Suite

After setting up your development environment, run tests.

```shell
python manage.py test django_ledger
```

## Screenshots

*   ![Django Ledger Entity Dashboard](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_entity_dashboard.png)
*   ![Django Ledger Balance Sheet](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_income_statement.png)
*   ![Django Ledger Income Statement](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_balance_sheet.png)
*   ![Django Ledger Bill](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_bill.png)
*   ![Django Ledger Invoice](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_invoice.png)

## Financial Statements Screenshots

*   ![Balance Sheet](https://django-ledger.us-east-1.linodeobjects.com/public/img/BalanceSheetStatement.png)
*   ![Income Statement](https://django-ledger.us-east-1.linodeobjects.com/public/img/IncomeStatement.png)
*   ![Cash Flow Statement](https://django-ledger.us-east-1.linodeobjects.com/public/img/CashFlowStatement.png)