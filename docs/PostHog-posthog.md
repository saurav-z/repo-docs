<p align="center">
  <img alt="PostHog Logo" src="https://user-images.githubusercontent.com/65415371/205059737-c8a4f836-4889-4654-902e-f302b187b6a0.png">
</p>

<p align="center">
  <a href='https://posthog.com/contributors'><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/posthog/posthog"/></a>
  <a href='http://makeapullrequest.com'><img alt='PRs Welcome' src='https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=shields'/></a>
  <img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/posthog/posthog"/>
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/posthog/posthog"/>
  <img alt="GitHub closed issues" src="https://img.shields.io/github/issues-closed/posthog/posthog"/>
</p>

<p align="center">
  <a href="https://posthog.com/docs">Docs</a> - <a href="https://posthog.com/community">Community</a> - <a href="https://posthog.com/roadmap">Roadmap</a> - <a href="https://posthog.com/why">Why PostHog?</a> - <a href="https://posthog.com/changelog">Changelog</a> - <a href="https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md">Bug reports</a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=2jQco8hEvTI">
    <img src="https://res.cloudinary.com/dmukukwp6/image/upload/demo_thumb_68d0d8d56d" alt="PostHog Demonstration">
  </a>
</p>

## PostHog: The Open-Source Product Analytics & Feature Management Platform

PostHog is a powerful open-source platform offering a complete suite of tools to build successful products, including:

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, data visualization, and SQL querying.
*   **Web Analytics:** Monitor web traffic, user sessions, conversions, web vitals, and revenue with a GA-like dashboard.
*   **Session Replays:** Watch real user sessions to diagnose issues and understand user behavior.
*   **Feature Flags:** Safely roll out features to select users or cohorts.
*   **Experiments:** Test changes and measure their impact on key metrics.
*   **Error Tracking:** Track errors, receive alerts, and resolve issues to improve your product.
*   **Surveys:** Gather user feedback using no-code survey templates or a custom survey builder.
*   **Data Warehouse:** Sync data from external tools and query it alongside your product data.
*   **Data Pipelines:** Run custom filters and transformations on incoming data, and send it to numerous tools or your data warehouse.
*   **LLM Observability:** Capture traces, generations, latency, and cost for your LLM-powered app.

**[View the PostHog source code on GitHub](https://github.com/PostHog/posthog)**

## Getting Started

### PostHog Cloud (Recommended)

Sign up for a free [PostHog Cloud](https://us.posthog.com/signup) account (or [EU](https://eu.posthog.com/signup)) for the fastest and most reliable start.  Enjoy a generous free tier!

### Self-hosting (Advanced)

Deploy a hobby instance with Docker on Linux:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

_Note: Open-source deployments are for smaller event volumes and are not supported. See the documentation for [Self-Hosting](https://posthog.com/docs/self-host)._

## Setting Up PostHog

Integrate PostHog with your project using the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), SDKs, or API.

**SDKs Available For:**

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Plus support for Go, .NET/C#, Django, Angular, WordPress, and Webflow.

## Learn More

*   [Company Handbook](https://posthog.com/handbook)
*   [Winning with PostHog Guide](https://posthog.com/docs/new-to-posthog/getting-hogpilled)

## Contributing

*   [Roadmap](https://posthog.com/roadmap)
*   [Developing Locally](https://posthog.com/handbook/engineering/developing-locally)
*   [Feature Request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md)
*   [Bug Report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open-Source vs. Paid

This repository is available under the [MIT Expat License](https://github.com/PostHog/posthog/blob/master/LICENSE), excluding the `ee` directory.

For a fully FOSS version, check out [posthog-foss](https://github.com/PostHog/posthog-foss).

[Pricing](https://posthog.com/pricing) is transparent.

## We're Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join the team!  [Apply to our open positions](https://posthog.com/careers).