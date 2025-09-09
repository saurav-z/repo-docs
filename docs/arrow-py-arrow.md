# Arrow: Human-Friendly Dates and Times for Python

**Simplify date and time manipulation in Python with Arrow, a user-friendly library that makes working with datetime objects a breeze.**  For more information, visit the original repository: [https://github.com/arrow-py/arrow](https://github.com/arrow-py/arrow).

## Why Choose Arrow?

Arrow offers a more intuitive and efficient way to work with dates, times, and timezones compared to Python's built-in modules. It eliminates the complexity of dealing with multiple modules and types.

## Key Features

*   **Simplified datetime replacement:** Easily create, manipulate, and format dates and times.
*   **Python Version Support:**  Compatible with Python 3.8 and later.
*   **Timezone-aware by Default:** Works with UTC for clarity and avoids timezone-related issues.
*   **Easy Creation:** Simple options for common input scenarios.
*   **Flexible Manipulation:** Use the `shift` method for relative offsets, including weeks.
*   **Automatic Formatting and Parsing:**  Effortlessly format and parse strings.
*   **ISO 8601 Support:**  Full support for the widely used ISO 8601 standard.
*   **Timezone Conversion:** Easily convert between timezones.
*   **Integration with Popular Libraries:** Supports `dateutil`, `pytz`, and `ZoneInfo`.
*   **Time Span Generation:** Generate time spans, ranges, floors, and ceilings for various time frames.
*   **Humanization:**  Humanize dates and times with locale support (e.g., "an hour ago," "한시간 전").
*   **Extensible:** Create your own Arrow-derived types.
*   **Type Hinting:** Full support for PEP 484-style type hints.

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
arrow.get('2013-05-11T21:23:58.970460+07:00')  # Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()  # Output: <Arrow [2024-10-27T15:30:00+00:00]>

# Shift the time
utc = utc.shift(hours=-1)  # Output: <Arrow [2024-10-27T14:30:00+00:00]>

# Convert to a different timezone
local = utc.to('US/Pacific')  # Output: <Arrow [2024-10-27T07:30:00-07:00]>

# Get a timestamp
local.timestamp()  # Output: 1703232300.0

# Format the time
local.format()  # Output: 2013-05-11 06:23:58 -07:00

# Format with a specific pattern
local.format('YYYY-MM-DD HH:mm:ss ZZ')  # Output: 2013-05-11 06:23:58 -07:00

# Humanize the time
local.humanize()  # Output: 14 hours ago

# Humanize with a specific locale
local.humanize(locale='ko-kr')  # Output: 14시간 전
```

## Documentation

For complete documentation, please visit [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome!  See the original README for contributing guidelines.

## Support Arrow

Consider supporting Arrow through the [Open Collective](https://opencollective.com/arrow).