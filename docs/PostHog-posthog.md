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

# PostHog: The Open-Source Product Analytics Platform

**PostHog empowers you to build successful products with a complete suite of tools, all in one open-source platform.**  Learn more about the original repository on [GitHub](https://github.com/PostHog/posthog).

## Key Features

*   **Product Analytics:** Understand user behavior with event-based analytics, data visualization, and SQL querying.
*   **Web Analytics:** Monitor website traffic, user sessions, conversions, web vitals, and revenue with a GA-like dashboard.
*   **Session Replays:** Watch real user sessions to diagnose issues and gain insights into user interactions.
*   **Feature Flags:** Safely roll out features to specific users or cohorts using feature flags.
*   **Experiments:** A/B test changes and measure their impact on goal metrics.
*   **Error Tracking:** Track and resolve errors with alerts to improve product quality.
*   **Surveys:** Gather user feedback using pre-built templates or a custom survey builder.
*   **Data Warehouse:** Integrate data from external tools like Stripe, Hubspot, and your data warehouse.
*   **Data Pipelines (CDP):** Transform and send data to 25+ tools or any webhook in real-time.
*   **LLM Analytics:** Capture traces, generations, latency, and cost for your LLM-powered apps.

## Getting Started

### PostHog Cloud (Recommended)

The easiest way to get started is by signing up for a free PostHog Cloud account: [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).  Enjoy a generous free tier with millions of events, recordings, and more, with usage-based pricing beyond the free tier.

### Self-Hosting (Advanced)

For self-hosting, deploy a hobby instance with Docker (recommended 4GB memory):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

*Note: Open-source deployments scale to approximately 100k events per month. For more information, see our [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer).*

## Setup and Integration

Integrate PostHog into your product using our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet) or one of our comprehensive SDKs.

### SDKs and Libraries

PostHog supports a wide range of languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

We also provide documentation for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

Detailed setup and usage guides are available in our [product docs](https://posthog.com/docs/product-os).

## Learn More

*   [Company Handbook](https://posthog.com/handbook): Explore our strategy, culture, and processes.
*   [Winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled): A guide to measuring activation, retention, and revenue.

## Contribute

We welcome contributions!

*   [Roadmap](https://posthog.com/roadmap): Vote on features and get early access to beta features.
*   Open a PR: Review our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Licensing

This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)) if applicable.

For a completely free and open-source option, check out [posthog-foss](https://github.com/PostHog/posthog-foss).

Transparent pricing is available on [our pricing page](https://posthog.com/pricing).

## We're Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join our growing team! Explore career opportunities at [PostHog](https://posthog.com/careers).