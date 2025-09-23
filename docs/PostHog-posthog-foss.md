<p align="center">
  <img alt="PostHog Logo" src="https://user-images.githubusercontent.com/65415371/205059737-c8a4f836-4889-4654-902e-f302b187b6a0.png">
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

## PostHog: The Open-Source Product OS for Data-Driven Success

PostHog is an all-in-one, open-source product analytics platform that gives you the tools to build successful products, from understanding user behavior to safely rolling out features. **Dive into the code on [GitHub](https://github.com/PostHog/posthog-foss) and start building today!**

**Key Features:**

*   📊 **Product Analytics:** Understand user behavior with event-based analytics, visualization, and SQL.
*   🌐 **Web Analytics:** Monitor web traffic, user sessions, and key metrics like conversion and revenue.
*   🎬 **Session Replays:** Watch real user sessions to diagnose issues and understand user interactions.
*   🚩 **Feature Flags:** Safely roll out features to specific users or cohorts.
*   🧪 **Experiments:** Test changes and measure their impact on goal metrics.
*   🐞 **Error Tracking:** Track and resolve errors to improve product quality.
*   💬 **Surveys:** Gather user feedback with customizable surveys.
*   📦 **Data Warehouse & Pipelines:** Sync data from external sources and transform your data.
*   💡 **LLM Analytics:** Capture traces, generations, latency, and cost for your LLM-powered applications.

## Getting Started

### PostHog Cloud (Recommended)

The easiest way to get started is by signing up for a free account on [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Enjoy a generous free tier!

### Self-hosting (Advanced)

Deploy a hobby instance with Docker:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

*   Self-hosting is recommended for advanced users, and support is limited. See [Self-hosting Docs](https://posthog.com/docs/self-host) for more details.

## Setting Up PostHog

Integrate PostHog with your product using our:

*   [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet)
*   [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks)
*   [API](https://posthog.com/docs/getting-started/install?tab=api)

SDKs are available for:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Plus many more integrations!

## Learn More

*   **Company Handbook:** Explore our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).
*   **Winning with PostHog:** Learn the basics of [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contribute

We welcome contributions of all sizes:

*   Vote on features or get early access to beta functionality in our [roadmap](https://posthog.com/roadmap)
*   Open a PR (see our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally))
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open Source vs. Paid

This repo is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)) if applicable.

For a truly open-source experience, check out [posthog-foss](https://github.com/PostHog/posthog-foss), which is purged of all proprietary code and features.

See our [pricing page](https://posthog.com/pricing) for transparent pricing of our paid plans.

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join the PostHog team! We're growing fast and looking for talented individuals. Apply at [https://posthog.com/careers](https://posthog.com/careers).