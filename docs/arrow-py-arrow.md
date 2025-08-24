# Arrow: Human-Friendly Dates and Times for Python

**Simplify your Python date and time manipulation with Arrow, a user-friendly library that makes working with dates, times, and timezones intuitive and efficient.**  [View the original repository on GitHub](https://github.com/arrow-py/arrow)

Arrow provides a more sensible and human-friendly approach to working with dates and times in Python, offering a drop-in replacement for the built-in `datetime` module with enhanced functionality.

## Key Features

*   **Intuitive and Readable:**  Simplified date and time creation, manipulation, and formatting.
*   **Timezone-Aware:**  Handles timezones and UTC by default, making conversions effortless.
*   **ISO 8601 Support:** Robust parsing and formatting for the ISO 8601 standard.
*   **Humanization:** Easily convert dates and times into human-readable formats (e.g., "an hour ago").
*   **Flexible Time Shifts:**  Shift dates and times by various units, including weeks, months, and years.
*   **Time Span Generation:** Generate time spans, ranges, floors and ceilings for time frames.
*   **Extensible:**  Customize and extend Arrow for your specific needs.

## Why Choose Arrow?

Arrow overcomes the usability shortcomings of Python's built-in `datetime` module and related libraries like `dateutil` and `pytz`. It simplifies common tasks, such as:

*   Reducing the number of modules and types needed.
*   Simplifying timezone and timestamp conversions.
*   Offering intuitive creation options.
*   Providing human-friendly date/time presentation.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Parse a datetime string
a = arrow.get('2013-05-11T21:23:58.970460+07:00')
print(a)  # Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get UTC time
utc = arrow.utcnow()
print(utc)  # Output: <Arrow [2024-01-01T00:00:00+00:00]> (example)

# Shift the time
utc = utc.shift(hours=-1)
print(utc) # Output: <Arrow [2023-12-31T23:00:00+00:00]> (example)

# Convert to a different timezone
local = utc.to('US/Pacific')
print(local) # Output: <Arrow [2023-12-31T15:00:00-08:00]> (example)

# Format the time
print(local.format('YYYY-MM-DD HH:mm:ss ZZ'))  # Output: 2023-12-31 15:00:00 -08:00 (example)

# Humanize the time
print(local.humanize())  # Output: an hour ago (example)
```

## Documentation

For detailed information and examples, please visit the official documentation at [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome! See the [issue tracker](https://github.com/arrow-py/arrow/issues) for tasks.
Steps:

1.  Fork the repository.
2.  Create a branch for your changes.
3.  Add tests.
4.  Run the tests using `tox && tox -e lint,docs` or `make build39 && make test && make lint`.
5.  Submit a pull request.

## Support Arrow

Support the project via the [Open Collective](https://opencollective.com/arrow) for financial contributions.