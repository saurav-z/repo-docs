# FastStream: Effortlessly Build Microservices with Event Streams

**Simplify event stream integration for your services and build powerful, data-driven microservices with FastStream.**

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)

[Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[Coverage](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[Downloads](https://www.pepy.tech/projects/faststream)
[PyPI](https://pypi.org/project/faststream)
[Python versions](https://img.shields.io/pypi/pyversions/faststream.svg)

[CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[License](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[Code of Conduct](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)

[Discord](https://discord.gg/qFm6aSqq59)
[FastStream Shield](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)
[Telegram](https://t.me/python_faststream)

[Gurubase](https://gurubase.io/g/faststream)

---

## Key Features of FastStream

FastStream is a Python framework that streamlines the development of microservices using event streams. Here's what makes it stand out:

*   **Multiple Broker Support:**  Seamlessly work with various message brokers like Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Integrate Pydantic for robust data validation and serialization of incoming messages.
*   **Automatic AsyncAPI Documentation:** Generate comprehensive documentation automatically, simplifying service integration.
*   **Intuitive Development:** Benefit from full-typed editor support, catching errors early in the development process.
*   **Powerful Dependency Injection:** Manage service dependencies efficiently with a built-in DI system.
*   **Testable Code:** Leverage in-memory testing capabilities to accelerate your CI/CD pipeline.
*   **Extensibility:** Utilize extensions for lifespans, custom serialization, and middleware customization.
*   **Framework Integrations:** Works seamlessly with any HTTP framework, with a focus on FastAPI integration.

For more details, see the [FastStream documentation](https://faststream.ag2.ai/latest/).

---

## Core Functionality

### Installation

Install FastStream using pip:

```bash
pip install 'faststream[kafka]'  # or [rabbit, nats, redis]
```

### Code Example

Build producers and consumers with easy-to-use decorators:

```python
from faststream import FastStream
from faststream.kafka import KafkaBroker

broker = KafkaBroker("localhost:9092")
app = FastStream(broker)

@broker.subscriber("in")
@broker.publisher("out")
async def handle_msg(user: str, user_id: int) -> str:
    return f"User: {user_id} - {user} registered"
```

### Testing

Test your services efficiently using `TestBroker`:

```python
import pytest
from faststream.kafka import TestKafkaBroker

@pytest.mark.asyncio
async def test_correct():
    async with TestKafkaBroker(broker) as br:
        await br.publish({"user": "John", "user_id": 1,}, "in")
```

### Running the Application

Use the FastStream CLI to run your application with features like hot-reloading and multiprocessing:

```bash
pip install "faststream[cli]"
faststream run basic:app
faststream run basic:app --reload
faststream run basic:app --workers 3
```

### Project Documentation

FastStream automatically generates documentation using the [AsyncAPI](https://www.asyncapi.com/) specification.  This makes it simple to understand the available channels and message formats.

---

## Additional Information

### History

FastStream evolved from [FastKafka](https://github.com/airtai/fastkafka) and [Propan](https://github.com/lancetnik/propan), incorporating the best features from both projects.

### Dependencies

FastStream utilizes [FastDepends](https://lancetnik.github.io/FastDepends/) for dependency management, similar to `pytest fixtures` and `FastAPI Depends`.

### HTTP Framework Integrations

FastStream is designed to integrate seamlessly with various HTTP frameworks, including:

*   **Any Framework:** Use `MQBrokers` independently or with your application lifespan.
*   **FastAPI Plugin:** Utilize `StreamRouter` for easy integration.

---

## Get Involved

*   Star the project on [GitHub](https://github.com/ag2ai/faststream/)
*   Join the [Discord server](https://discord.gg/qFm6aSqq59)
*   Join the [Telegram group](https://t.me/python_faststream)

---

## Contributors

[Contributors](https://github.com/ag2ai/faststream/graphs/contributors)