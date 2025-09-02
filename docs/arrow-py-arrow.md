# Arrow: Effortless Date and Time Handling in Python

**Simplify date and time manipulation in Python with Arrow, a user-friendly library designed to make working with dates, times, and timestamps a breeze.** ([View on GitHub](https://github.com/arrow-py/arrow))

Arrow provides a more intuitive and efficient way to handle dates and times, offering a drop-in replacement for Python's built-in `datetime` module with enhanced functionality and ease of use.

## Key Features

*   **User-Friendly API:** Simplifies date and time creation, manipulation, and formatting.
*   **Timezone Awareness:** Handles timezones seamlessly, defaulting to UTC.
*   **ISO 8601 Support:** Comprehensive support for the ISO 8601 standard for parsing and formatting.
*   **Humanization:**  Easily humanize dates and times for more readable output (e.g., "an hour ago").
*   **Flexible Shifting:** Use the `shift` method with relative offsets, including weeks.
*   **Time Span Generation:** Generates time spans, ranges, floors, and ceilings for diverse timeframes.
*   **Extensible:**  Supports custom Arrow-derived types and PEP 484-style type hints.
*   **Python Compatibility:** Supports Python 3.8+.

## Why Choose Arrow?

Arrow streamlines date and time operations, addressing the usability issues of Python's standard library:

*   **Reduces Complexity:** Eliminates the need to juggle multiple modules and types.
*   **Simplifies Timezone Handling:** Makes timezone conversions straightforward and less verbose.
*   **Enhances Functionality:** Includes features like ISO 8601 parsing, humanization, and time span generation.

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
arrow.get('2013-05-11T21:23:58.970460+07:00')
# Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get UTC now
utc = arrow.utcnow()
# Output: <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift time
utc = utc.shift(hours=-1)
# Output: <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to a different timezone
local = utc.to('US/Pacific')
# Output: <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get timestamp
local.timestamp()
# Output: 1368303838.970460

# Format date and time
local.format()
# Output: '2013-05-11 13:23:58 -07:00'
local.format('YYYY-MM-DD HH:mm:ss ZZ')
# Output: '2013-05-11 13:23:58 -07:00'

# Humanize the date
local.humanize()
# Output: 'an hour ago'
local.humanize(locale='ko-kr')
# Output: '한시간 전'
```

## Documentation

For in-depth information and examples, visit the official documentation at [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

We welcome contributions! To contribute:

1.  Find an issue or feature to address on the [issue tracker](https://github.com/arrow-py/arrow/issues).
2.  Fork the repository and create a branch for your changes.
3.  Add tests to cover your changes.
4.  Run the test suite and linting checks: `tox && tox -e lint,docs` or `make build39 && make test && make lint` (adjust Python version as needed).
5.  Submit a pull request.

Have questions?  Ask them in the [discussions](https://github.com/arrow-py/arrow/discussions).

## Support Arrow

Consider supporting the project through [Open Collective](https://opencollective.com/arrow) to help ensure its continued development and maintenance.