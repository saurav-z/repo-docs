<p align="center">
  <img alt="posthoglogo" src="https://user-images.githubusercontent.com/65415371/205059737-c8a4f836-4889-4654-902e-f302b187b6a0.png">
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

# PostHog: Open-Source Product Analytics and More 🚀

**PostHog is the all-in-one, open-source platform empowering you to build successful products by providing powerful product analytics, session replays, feature flags, and more.**  [Explore the source code on GitHub](https://github.com/PostHog/posthog-foss).

## Key Features

*   **Product Analytics:** Understand user behavior through event-based analytics, data visualization, and SQL queries.
*   **Web Analytics:**  Monitor web traffic, user sessions, conversion rates, web vitals, and revenue with a GA-like dashboard.
*   **Session Replays:**  Watch real user sessions to identify issues and gain deeper insights into user interactions.
*   **Feature Flags:**  Safely roll out new features to specific user segments.
*   **Experiments:**  A/B test changes and measure their impact on key metrics with a no-code setup.
*   **Error Tracking:**  Track errors, receive alerts, and resolve issues to improve your product's reliability.
*   **Surveys:**  Gather user feedback with no-code survey templates and a custom survey builder.
*   **Data Warehouse:**  Integrate data from external tools, such as Stripe and Hubspot, and query it alongside your product data.
*   **Data Pipelines:**  Run custom filters and transformations on incoming data and send it to various tools or a webhook in real time or batch export large amounts to your warehouse.
*   **LLM Analytics:** Capture traces, generations, latency, and cost for your LLM-powered app.

## Getting Started

PostHog offers two primary deployment options:

### PostHog Cloud (Recommended)

The easiest way to get started is by signing up for a free [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) account.  Enjoy a generous free tier to get you started.

### Self-hosting (Advanced)

For self-hosting, you can deploy a hobby instance with Docker:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

*   Self-hosted deployments are recommended for up to 100k events per month.

## Setting Up PostHog

Integrate PostHog with your product using:

*   [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet)
*   SDKs:  Explore SDKs for various platforms:

    *   Frontend: JavaScript, React Native, Next.js, React, Vue
    *   Mobile: Android, iOS, Flutter
    *   Backend: Python, Node, PHP, Ruby
    *   Additional libraries for Go, .NET/C#, Django, Angular, WordPress, and Webflow.

*   [API](https://posthog.com/docs/getting-started/install?tab=api).

  Refer to the [product docs](https://posthog.com/docs/product-os) for guidance on setting up different features.

## Learn More

*   [Company Handbook](https://posthog.com/handbook):  Discover PostHog's strategy, culture, and processes.
*   [Winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled):  A guide for maximizing PostHog's capabilities.
*   [Product Documentation](https://posthog.com/docs)

## Contribute

We welcome contributions!

*   Vote on features or access beta functionality in our [roadmap](https://posthog.com/roadmap).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).
*   Submit a PR (see instructions in the [handbook](https://posthog.com/handbook/engineering/developing-locally)).

## Open Source vs. Paid

This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), excluding the `ee` directory, which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE).

For a completely FOSS option, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.  See the [pricing page](https://posthog.com/pricing) for the details of our paid plans.

## We're Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Interested in joining the team?  We're growing fast!  Check out our [careers page](https://posthog.com/careers).