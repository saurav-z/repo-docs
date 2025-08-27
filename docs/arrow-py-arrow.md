# Arrow: Effortless Date and Time Handling in Python

Arrow is a Python library that simplifies working with dates, times, and timestamps, making your code cleaner and more readable. ([See the original repo](https://github.com/arrow-py/arrow)).

## Key Features

*   **Intuitive Date & Time Creation:** Easily create Arrow objects from various input formats.
*   **Timezone-Aware by Default:**  Works seamlessly with timezones, handling UTC by default.
*   **Simplified Manipulation:**  Use the `shift` method for intuitive date and time adjustments (e.g., days, hours, weeks).
*   **Automatic Formatting & Parsing:**  Parse and format dates/times effortlessly, including robust ISO 8601 support.
*   **Human-Friendly Output:**  Get humanized representations of dates and times (e.g., "an hour ago").
*   **Time Span Capabilities**: Generate time spans, ranges, floors and ceilings.
*   **Localization Support:**  Humanize dates in multiple locales.
*   **Type Hinting**: Full support for PEP 484-style type hints

## Why Choose Arrow?

Arrow addresses the complexities of Python's built-in datetime modules, offering a more user-friendly and efficient approach:

*   **Fewer Modules:** Avoid juggling multiple modules like `datetime`, `time`, and `dateutil`.
*   **Simplified Types:**  Work with a more streamlined set of date and time object types.
*   **Easy Timezone Handling:**  Simplify timezone conversions and timestamp management.
*   **Comprehensive Functionality:**  Access features like ISO 8601 parsing, time spans, and humanization that are missing from the standard library.

## Quick Start

### Installation

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Create an Arrow object
dt = arrow.get('2023-10-27T10:00:00+00:00')
print(dt)  # Output: <Arrow [2023-10-27T10:00:00+00:00]>

# Shift the time
dt_shifted = dt.shift(hours=2)
print(dt_shifted)  # Output: <Arrow [2023-10-27T12:00:00+00:00]>

# Convert to a different timezone
dt_local = dt_shifted.to('US/Pacific')
print(dt_local) # Output: <Arrow [2023-10-27T05:00:00-07:00]>

# Format the date
print(dt_local.format('YYYY-MM-DD HH:mm:ss ZZ'))  # Output: 2023-10-27 05:00:00 -07:00

# Humanize the date
print(dt_local.humanize()) # Output: an hour ago
```

## Documentation

For complete documentation, see [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome!  Find issues and help improve Arrow. For more details, see the original README or the documentation.

## Support Arrow

Support the project by making a financial contribution on the `Open Collective <https://opencollective.com/arrow>`_.