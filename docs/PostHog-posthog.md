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

## PostHog: The Open Source Platform for Product-Led Growth

**PostHog is your all-in-one, open-source powerhouse for building successful products, providing a comprehensive suite of tools for product analytics, user behavior analysis, and feature management.** ([See the original repo](https://github.com/PostHog/posthog))

### Key Features

*   **Product Analytics:** Understand user behavior with event-based analytics, including autocapture and manual instrumentation.
*   **Web Analytics:** Monitor web traffic, user sessions, and key metrics like conversion and revenue, similar to Google Analytics.
*   **Session Replays:** Watch real user sessions to diagnose issues and gain deeper insights into user interactions.
*   **Feature Flags:** Safely roll out features to specific users or cohorts.
*   **Experiments:** A/B test changes and measure their impact on goals without code.
*   **Error Tracking:**  Track errors, receive alerts, and resolve issues to improve product stability.
*   **Surveys:** Gather user feedback with no-code survey templates or a custom survey builder.
*   **Data Warehouse Integration:** Sync data from external tools (Stripe, Hubspot, etc.) and query it alongside your product data.
*   **Data Pipelines (CDP):** Customize filters and transformations on incoming data, and send it to 25+ tools in real time or export to your warehouse.
*   **LLM Observability:** Monitor LLM-powered applications by capturing traces, generations, latency, and cost.

### Getting Started

#### PostHog Cloud (Recommended)

The easiest and most reliable way to get started with PostHog is to sign up for free at [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).  Enjoy a generous free tier!

#### Self-hosting (Advanced)

For self-hosting, deploy a hobby instance with Docker:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

_Note: Open source deployments are not supported. Check our self-hosting docs, troubleshooting guide, and disclaimer for more info._

### Setting Up PostHog

Integrate PostHog into your project using the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or the [API](https://posthog.com/docs/getting-started/install?tab=api).

**SDKs Available:**

*   **Frontend:** JavaScript, Next.js, React, Vue
*   **Mobile:** React Native, Android, iOS, Flutter
*   **Backend:** Python, Node, PHP, Ruby

**Further Documentation:**

*   [Product Analytics](https://posthog.com/docs/product-analytics/capture-events)
*   [Web Analytics](https://posthog.com/docs/web-analytics/getting-started)
*   [Session Replay](https://posthog.com/docs/session-replay/how-to-watch-recordings)
*   [Feature Flags](https://posthog.com/docs/feature-flags/creating-feature-flags)
*   [Experiments](https://posthog.com/docs/experiments/creating-an-experiment)
*   [Error Tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture)
*   [Surveys](https://posthog.com/docs/surveys/installation)
*   [Data Warehouse](https://posthog.com/docs/cdp/sources)

### Learning More

*   [Company Handbook](https://posthog.com/handbook) (Open-sourced!)
*   [Winning with PostHog Guide](https://posthog.com/docs/new-to-posthog/getting-hogpilled)

### Contributing

We welcome contributions!

*   [Roadmap](https://posthog.com/roadmap)
*   Open a PR (see instructions on developing PostHog locally)
*   [Feature Request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md)
*   [Bug Report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

### Open-Source vs. Paid

*   This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), excluding the `ee` directory (see its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)).
*   For a truly FOSS solution, check out [posthog-foss](https://github.com/PostHog/posthog-foss).
*   [Pricing page](https://posthog.com/pricing)

### We're Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join our growing team!  See our [careers page](https://posthog.com/careers).