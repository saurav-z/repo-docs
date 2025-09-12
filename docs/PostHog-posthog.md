<p align="center">
  <img alt="posthoglogo" src="https://user-images.githubusercontent.com/65415371/205059737-c8a4f836-4889-4654-902e-f302b160.png">
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

# PostHog: Open Source Product Analytics, Session Replay, and More

**Supercharge your product development with PostHog, the all-in-one, open-source platform for product analytics, session replays, feature flags, and more.** ([View on GitHub](https://github.com/PostHog/posthog))

## Key Features

*   **Product Analytics:** Understand user behavior with event-based analytics, visualizations, and SQL querying.
*   **Web Analytics:** Monitor website traffic, user sessions, conversions, and web vitals with ease.
*   **Session Replays:** Watch real user sessions to diagnose issues and understand user behavior.
*   **Feature Flags:** Safely roll out features to specific user cohorts.
*   **Experiments:** Test changes and measure their impact on key metrics using A/B tests with no-code setup.
*   **Error Tracking:** Track errors, get alerts, and resolve issues to improve your product.
*   **Surveys:** Collect user feedback with no-code survey templates or custom survey builders.
*   **Data Warehouse:** Sync data from external tools and query it alongside your product data.
*   **Data Pipelines:** Run custom filters and transformations on your data and send it to over 25+ tools.
*   **LLM Analytics:** Capture traces, generations, latency, and cost for your LLM-powered app.

## Getting Started

Choose the best option for you:

### PostHog Cloud (Recommended)

Get started quickly with a free tier and simple setup. Sign up at [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

### Self-hosting the open-source hobby deploy (Advanced)

Deploy a hobby instance on Linux with Docker:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

For detailed information on self-hosting, consult the [self-hosting docs](https://posthog.com/docs/self-host).

## Setting Up PostHog

Integrate PostHog with your applications using our flexible options:

*   [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet)
*   [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks) for various languages and frameworks
*   [API](https://posthog.com/docs/getting-started/install?tab=api)

We provide SDKs for:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Plus comprehensive docs for: Go, .NET/C#, Django, Angular, WordPress, Webflow, and more.

Once installed, explore our product docs for: product analytics, web analytics, session replays, feature flags, experiments, error tracking, surveys, and more.

## Learning More

*   [Company Handbook](https://posthog.com/handbook)
*   [Winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled)

## Contributing

We welcome contributions!

*   [Roadmap](https://posthog.com/roadmap)
*   [Developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)
*   [Feature Request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md)
*   [Bug Report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open-Source vs. Paid

This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE). The `ee` directory has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE). For a fully open-source version without any proprietary code, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

View our [pricing page](https://posthog.com/pricing) for transparent information.

## We’re Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join our fast-growing team: [Careers at PostHog](https://posthog.com/careers).