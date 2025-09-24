<p align="center">
  <img alt="PostHog Logo" src="https://user-images.githubusercontent.com/65415371/205059737-c8a4f836-4889-4654-902e-f302b160.png">
</p>

<p align="center">
  <a href='https://posthog.com/contributors'><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/posthog/posthog"/></a>
  <a href='http://makeapullrequest.com'><img alt='PRs Welcome' src='https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=shields'/></a>
  <img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/posthog/posthog"/>
  <a href="https://github.com/PostHog/posthog/commits/master"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/posthog/posthog"/> </a>
  <a href="https://github.com/PostHog/posthog/issues?q=is%3Aissue%20state%3Aclosed"><img alt="GitHub closed issues" src="https://img.shields.io/github/issues-closed/posthog/posthog"/> </a>
</p>

<p align="center">
  <a href="https://posthog.com/docs">Docs</a> - <a href="https://posthog.com/community">Community</a> - <a href="https://posthog.com/roadmap">Roadmap</a> - <a href="https://posthog.com/why">Why PostHog?</a> - <a href="https://posthog.com/changelog">Changelog</a> - <a href="https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md">Bug reports</a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=2jQco8hEvTI">
    <img src="https://res.cloudinary.com/dmukukwp6/image/upload/demo_thumb_68d0d8d56d" alt="PostHog Demonstration">
  </a>
</p>

# PostHog: Open-Source Product Analytics for Building Successful Products

**PostHog** is an all-in-one, open-source product analytics platform that helps you understand user behavior, ship better products, and drive growth. ([Back to the original repository](https://github.com/PostHog/posthog))

## Key Features

*   **Product Analytics:** Understand user behavior with event-based analytics, visualizations, and SQL queries.
*   **Web Analytics:** Monitor website traffic, user sessions, conversion rates, and web vitals with a GA-like dashboard.
*   **Session Replays:** Watch real user sessions to diagnose issues and understand user interactions.
*   **Feature Flags:** Safely roll out new features to specific user segments.
*   **Experiments:** Run A/B tests and measure the impact of changes on your key metrics.
*   **Error Tracking:** Track errors, get alerts, and resolve issues quickly.
*   **Surveys:** Gather user feedback with customizable surveys.
*   **Data Warehouse:** Integrate data from external tools and analyze it alongside your product data.
*   **Data Pipelines:** Transform and route your data to 25+ tools in real-time.
*   **LLM Analytics:** Capture traces, generations, latency, and cost for your LLM-powered applications.

## Getting Started

### PostHog Cloud (Recommended)

The easiest way to get started is by signing up for a free account on [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Enjoy a generous free tier with 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 1500 survey responses monthly!

### Self-Hosting (Advanced)

For self-hosting, deploy a hobby instance with Docker:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Learn more in our [self-hosting docs](https://posthog.com/docs/self-host).

## Setting Up PostHog

Integrate PostHog into your project using our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet) or one of our comprehensive [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks) for various languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

We also offer documentation for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

## Learn More

Explore our open-source [company handbook](https://posthog.com/handbook) to learn about our culture, strategy, and processes. Check out our guide to [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled) for measuring activation, retention, and revenue.

## Contributing

We welcome contributions!

*   [Roadmap](https://posthog.com/roadmap)
*   [Developing PostHog Locally](https://posthog.com/handbook/engineering/developing-locally))
*   [Feature Request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md)
*   [Bug Report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open-Source vs. Paid

PostHog is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)) if applicable. For a completely open-source version without any proprietary code, see our [posthog-foss](https://github.com/PostHog/posthog-foss) repository. View our [pricing page](https://posthog.com/pricing) for details on our paid plans.

## We're Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join our growing team! Explore [our careers](https://posthog.com/careers).