<p align="center">
  <img alt="PostHog Logo" src="https://user-images.githubusercontent.com/65415371/205059737-c8a4f836-4889-4654-902e-f302b160.png">
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

## PostHog: Open-Source Product Analytics and Customer Data Platform (CDP)

**PostHog empowers product teams to build better products with a complete, open-source platform for product analytics, session replay, feature flags, and more.**  This repository hosts the core code for PostHog, allowing you to understand your users, track product performance, and iterate faster.  Explore the power of PostHog on [GitHub](https://github.com/PostHog/posthog).

### Key Features:

*   **Product Analytics:** Understand user behavior with event-based analytics, custom dashboards, and SQL querying.
*   **Web Analytics:** Monitor website traffic, conversion, and revenue with a GA-like dashboard.
*   **Session Replays:** Watch real user sessions to diagnose issues and understand user interactions.
*   **Feature Flags:**  Safely roll out new features to specific user segments.
*   **Experiments:**  A/B test changes and measure their impact on key metrics.
*   **Error Tracking:** Identify and resolve errors to improve product quality.
*   **Surveys:** Gather user feedback with customizable surveys.
*   **Data Warehouse & Pipelines:** Sync data from external tools, transform it, and send it to other tools.
*   **LLM Analytics:**  Capture traces, generations, latency, and cost for your LLM-powered app.

PostHog offers a [generous monthly free tier](https://posthog.com/pricing) and is free to use for its core features.

## Getting Started

### 1. PostHog Cloud (Recommended)

The easiest way to start is with a free account on [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).  Get started with up to 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 1500 survey responses free every month.

### 2. Self-Hosting (Advanced)

For self-hosting, deploy a hobby instance with Docker:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Self-hosted instances are suitable for up to approximately 100k events per month. Consider [migrating to PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud) as your usage grows.  Refer to the [self-hosting docs](https://posthog.com/docs/self-host) and [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting) for details.

## Setup and Integration

Integrate PostHog into your project using our SDKs and APIs.

### SDKs and Libraries:

*   **Frontend:** JavaScript, Next.js, React, Vue.js
*   **Mobile:** React Native, Android, iOS, Flutter
*   **Backend:** Python, Node.js, PHP, Ruby, Go, .NET/C#, Django, Angular, WordPress, Webflow, and more.

Find installation guides and examples in the [PostHog documentation](https://posthog.com/docs/getting-started/install).

## Resources

*   [Product Docs](https://posthog.com/docs/product-os)
*   [Winning with PostHog Guide](https://posthog.com/docs/new-to-posthog/getting-hogpilled)
*   [Company Handbook](https://posthog.com/handbook)

## Contributing

We welcome contributions of all sizes!

*   Contribute to our [roadmap](https://posthog.com/roadmap)
*   Submit a Pull Request (PR) -  see instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open Source vs. Paid

This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), with the exception of the `ee` directory which has its own [license](https://github.com/PostHog/posthog/blob/master/ee/LICENSE). For a completely free and open-source version, check out [posthog-foss](https://github.com/PostHog/posthog-foss). See our [pricing page](https://posthog.com/pricing) for transparent paid plan details.

## Join the Team

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

We're hiring!  If you're passionate about product analytics and open-source, [join our team](https://posthog.com/careers).