[![Django Ledger Logo](https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png)](https://github.com/arrobalytics/django-ledger)

# Django Ledger: Financial Management for Django

**Django Ledger** is a comprehensive, open-source accounting engine designed to bring robust financial management capabilities to your Django applications.

[Get Started Guide](https://www.djangoledger.com/get-started) | [Join our Discord](https://discord.gg/c7PZcbYgrc) | [Documentation](https://django-ledger.readthedocs.io/en/latest/) | [QuickStart Notebook](https://github.com/arrobalytics/django-ledger/blob/develop/notebooks/QuickStart%20Notebook.ipynb)

## Key Features of Django Ledger

*   **Double-Entry Accounting:** Ensures accuracy and provides a solid foundation for financial reporting.
*   **Hierarchical Chart of Accounts:** Organize your financial data with flexibility and clarity.
*   **Financial Statements:** Generate essential reports including Income Statements, Balance Sheets, and Cash Flow statements.
*   **Transaction Management:** Manage Ledgers, Journal Entries, and Transactions efficiently.
*   **Order and Invoice Processing:** Includes Purchase Orders, Sales Orders, Bills, and Invoices for streamlined workflows.
*   **Financial Ratio Calculations:** Gain insights with built-in financial analysis tools.
*   **Multi-Tenancy Support:** Allows you to manage multiple businesses or entities within a single application.
*   **Data Import/Export:** Supports OFX & QFX file import for easy integration with other financial systems.
*   **Inventory Management:** Track stock levels and manage inventory with ease.
*   **Built-in UI for Entity Management:** Simplify entity creation, modification, and management within your Django admin.
*   **Unit of Measures:** Allows for better tracking and organization of your inventory.
*   **Bank Account Information:** Store and use bank account information.
*   **Django Admin Integration:** Seamlessly integrated into your Django admin interface for easy access.
*   **Closing Entries:** Automated processes for period-end accounting.

## Getting Involved & Contributing

We welcome contributions of all kinds to help improve Django Ledger!

*   **Feature Requests/Bug Reports:** Open an issue in the [GitHub repository](https://github.com/arrobalytics/django-ledger).
*   **For Customization & Consulting:** [Contact us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com
*   **Contribute Code:** Review our [contribution guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md) and submit a pull request!

### Who Should Contribute?

We value contributions from developers with:

*   Strong Python and Django skills.
*   Experience in finance and accounting.
*   A passion for building a powerful accounting API.

## Installation & Setup

**Prerequisites:** A working Django project and basic Django knowledge are required.

You can use the zero-config Django Ledger starter template ([django-ledger-starter](https://github.com/arrobalytics/django-ledger-starter)).

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
                    'django_ledger.context.django_ledger_context'  # Add this line.
                ],
            },
        },
    ]
    ```

3.  **Perform Database Migrations:**

    ```bash
    python manage.py migrate
    ```

4.  **Add URLs to your `urls.py`:**

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

6.  **Access Django Ledger:** Navigate to the URL you defined (e.g., `http://127.0.0.1:8000/ledger`) in your browser and log in with your superuser credentials.

### Deprecated Behavior Setting

From v0.8.0+, manage deprecated features with the `DJANGO_LEDGER_USE_DEPRECATED_BEHAVIOR` setting in your Django settings file.
-   Default: `False`
-   To use deprecated features, set to `True`

## Development Setup

### Option 1: Using Docker (Recommended)

1.  Navigate to your project directory.
2.  Give executable permissions to `entrypoint.sh`:

    ```bash
    sudo chmod +x entrypoint.sh
    ```

3.  Add host `0.0.0.0` into `ALLOWED_HOSTS` in `settings.py`.
4.  Build and run the container:

    ```bash
    docker compose up --build
    ```

5.  Create a Django Superuser:
    *   Open a separate terminal and find the container ID: `docker ps`
    *   Execute this command, replacing `containerId` with the actual ID from the previous step:

    ```bash
    docker exec -it <containerId> /bin/sh
    ```

    *   Inside the container's shell, run: `python manage.py createsuperuser`

6.  Access Django Ledger in your browser at `http://0.0.0.0:8000/`.

### Option 2: Local Development Environment

1.  Clone the repository:

    ```bash
    git clone https://github.com/arrobalytics/django-ledger.git && cd django-ledger
    ```

2.  Install PipEnv (if not already installed):

    ```bash
    pip install -U pipenv
    ```

3.  Create a virtual environment:

    ```bash
    pipenv install
    ```

    (Optionally, specify Python version: `pipenv install --python /path/to/python`)

4.  Activate the environment:

    ```bash
    pipenv shell
    ```

5.  Apply database migrations:

    ```bash
    python manage.py migrate
    ```

6.  Create a Django superuser:

    ```bash
    python manage.py createsuperuser
    ```

7.  Run the development server:

    ```bash
    python manage.py runserver
    ```

## Running Tests

```bash
python manage.py test django_ledger
```

## Screenshots

[Include screenshots here, as in the original README, linking directly to the images]

---

**[View the Django Ledger GitHub Repository](https://github.com/arrobalytics/django-ledger)**