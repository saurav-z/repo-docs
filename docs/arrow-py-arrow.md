# Arrow: Human-Friendly Dates and Times for Python

**Simplify your Python date and time manipulation with Arrow, a powerful library that makes working with dates and times a breeze.**  For more details, explore the official repository: [https://github.com/arrow-py/arrow](https://github.com/arrow-py/arrow)

## Key Features:

*   **Intuitive and Readable:**  Arrow offers a human-friendly API for creating, manipulating, and formatting dates and times.
*   **Timezone-Aware by Default:**  Seamlessly handle timezones and UTC conversions.
*   **Simplified Creation:**  Easy-to-use options for creating date/time objects from various input formats.
*   **Flexible Time Shifting:**  Utilize the `shift` method for relative offsets (including weeks).
*   **Automatic Formatting and Parsing:**  Intelligent string parsing and formatting, with wide support for ISO 8601.
*   **Timezone Conversion:** Effortlessly convert between timezones.
*   **Broad Compatibility:**  Supports `dateutil`, `pytz`, and `ZoneInfo` tzinfo objects.
*   **Time Span Capabilities:** Generate time spans, ranges, floors, and ceilings.
*   **Humanization:** Human-readable date and time representations (e.g., "an hour ago").
*   **Extensible:**  Create your own Arrow-derived types.
*   **Type Hints:**  Full support for PEP 484-style type hints.

## Why Use Arrow?

Arrow overcomes the usability challenges of Python's built-in date/time modules by:

*   Consolidating multiple modules (datetime, time, etc.) into a single, unified interface.
*   Simplifying complex date/time object types (date, datetime, timedelta, etc.).
*   Making timezone and timestamp conversions easier to use.
*   Providing a more intuitive API, improving code readability.

## Quick Start

**Installation:**

```bash
pip install -U arrow
```

**Example Usage:**

```python
import arrow

# Create an Arrow object from an ISO 8601 string
now = arrow.get('2024-11-02T10:00:00+00:00')
print(now) # Output: <Arrow [2024-11-02T10:00:00+00:00]>

# Shift the time by a certain amount
past = now.shift(days=-2)
print(past) # Output: <Arrow [2024-10-31T10:00:00+00:00]>

# Convert to a different timezone
local = now.to('US/Pacific')
print(local) # Output: <Arrow [2024-11-02T02:00:00-08:00]>

# Format the date and time
print(local.format('YYYY-MM-DD HH:mm:ss ZZ')) # Output: 2024-11-02 02:00:00 -08:00

# Humanize the date
print(past.humanize()) # Output: 2 days ago
```

## Documentation

For comprehensive documentation, please visit [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome! See the original README for details.