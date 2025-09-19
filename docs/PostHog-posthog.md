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

# PostHog: The Open Source Product Analytics Platform

**PostHog is an open-source product analytics platform that empowers you to build better products by understanding user behavior.**  Dive into the [PostHog](https://github.com/PostHog/posthog) source code!

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, visualization, and SQL.
*   **Web Analytics:** Monitor web traffic, conversion, and revenue with a GA-like dashboard.
*   **Session Replays:** Watch real user sessions to diagnose issues and understand user behavior.
*   **Feature Flags:** Safely roll out features to specific users or cohorts.
*   **Experiments:** Test changes and measure their impact using no-code experimentation tools.
*   **Error Tracking:** Identify and resolve errors to improve product quality.
*   **Surveys:** Gather user feedback with a no-code survey builder.
*   **Data Warehouse & CDP:** Integrate data from external tools for comprehensive analysis, and run custom filters/transformations on your incoming data, sending it to 25+ tools.
*   **LLM Analytics:** Capture traces, generations, latency, and cost for your LLM-powered apps.

PostHog offers a [generous free tier](https://posthog.com/pricing), so you can start for free.

## Getting Started

### PostHog Cloud (Recommended)

The fastest and most reliable way to get started is by signing up for free on [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

### Self-Hosting (Advanced)

Deploy a hobby instance with Docker (4GB memory recommended):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Refer to the [self-hosting docs](https://posthog.com/docs/self-host) and [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting) for more details.

## Setting Up PostHog

Integrate PostHog using our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet) or one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks).

**SDKs Available For:**

*   **Frontend:** JavaScript, Next.js, React, Vue.js
*   **Mobile:** React Native, Android, iOS, Flutter
*   **Backend:** Python, Node, PHP, Ruby

Detailed guides are available for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

## Learning More

Explore our [company handbook](https://posthog.com/handbook) for insights into our strategy, culture, and processes. Learn how to maximize PostHog with our guide to [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled).

## Contributing

We welcome contributions of all sizes!

*   Vote on features and get early access to betas on our [roadmap](https://posthog.com/roadmap).
*   Submit a PR (see our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)).
*   Suggest a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-Source vs. Paid

This repository is licensed under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)) if applicable.

For a completely free and open-source option, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository. See our [pricing page](https://posthog.com/pricing) for transparent pricing.

## We're Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join our growing team! See our [careers page](https://posthog.com/careers) for open positions.