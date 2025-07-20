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

##  PostHog: The Open-Source Platform for Product Success

**PostHog** is an open-source product analytics platform providing powerful tools to help you build better products. ([Check out the original repo](https://github.com/PostHog/posthog))

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, visualizations, and SQL queries.
*   **Web Analytics:** Monitor web traffic, user sessions, conversion, web vitals, and revenue with a GA-like dashboard.
*   **Session Replays:** Watch real user sessions to diagnose issues and understand user interactions.
*   **Feature Flags:** Safely roll out features to specific users or cohorts using feature flags.
*   **Experiments (A/B Testing):** Test changes and measure their impact on key metrics.
*   **Error Tracking:** Track errors, get alerts, and resolve issues to improve your product.
*   **Surveys:** Gather user feedback with our no-code survey templates or build custom surveys.
*   **Data Warehouse:** Sync data from external tools to query it alongside your product data.
*   **Data Pipelines (CDP):** Transform and send data to 25+ tools or any webhook in real-time or batch export to your warehouse.
*   **LLM Observability:** Capture traces, generations, latency, and cost for your LLM-powered app.

Get started for free with our [generous monthly free tier](https://posthog.com/pricing). Sign up for [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

## Table of Contents

*   [PostHog: The Open-Source Platform for Product Success](#posthog-the-open-source-platform-for-product-success)
*   [Key Features](#key-features)
*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting up PostHog](#setting-up-posthog)
*   [Learning More about PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-source vs. paid](#open-source-vs-paid)
*   [We’re hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The easiest way to get started is to sign up for a free PostHog Cloud account: [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

*   Free tier includes: 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses/month.

### Self-hosting the open-source hobby deploy (Advanced)

Deploy a hobby instance on Linux with Docker (4GB memory recommended):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

*   Self-hosted instances should scale to ~100k events/month, then we recommend [migrating to PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud).
*   No customer support or guarantees for open-source deployments.
*   See our [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer).

## Setting up PostHog

Integrate PostHog with your project using our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or the [API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs are available for:

*   **Frontend:** JavaScript, Next.js, React, Vue
*   **Mobile:** React Native, Android, iOS, Flutter
*   **Backend:** Python, Node, PHP, Ruby

See our product docs ([product docs](https://posthog.com/docs/product-os)) for:

*   [Product analytics](https://posthog.com/docs/product-analytics/capture-events)
*   [Web analytics](https://posthog.com/docs/web-analytics/getting-started)
*   [Session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings)
*   [Feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags)
*   [Experiments](https://posthog.com/docs/experiments/creating-an-experiment)
*   [Error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture)
*   [Surveys](https://posthog.com/docs/surveys/installation)
*   [Data warehouse](https://posthog.com/docs/cdp/sources)

## Learning More about PostHog

*   Explore our open-source [company handbook](https://posthog.com/handbook) to learn more about our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [culture](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).
*   Check out our guide on [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled) to measure activation, track retention, and capture revenue.

## Contributing

We welcome contributions!

*   Vote on features in our [roadmap](https://posthog.com/roadmap)
*   Submit a PR ([developing locally](https://posthog.com/handbook/engineering/developing-locally))
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open-source vs. paid

*   This repo is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), with the exception of the `ee` directory.
*   For 100% FOSS, see [posthog-foss](https://github.com/PostHog/posthog-foss).
*   Transparent pricing is available on [our pricing page](https://posthog.com/pricing).

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Join our growing team! See our open roles and [apply here](https://posthog.com/careers).
```

Key improvements and SEO considerations:

*   **Clear Headline:** "PostHog: The Open-Source Platform for Product Success" uses a keyword and highlights the core benefit.
*   **Concise Summary:**  The first sentence is a strong hook and describes the platform.
*   **Bulleted Key Features:**  Easy for readers to scan and quickly understand what PostHog offers.
*   **Descriptive Subheadings:** Help with readability and SEO by organizing content and including keywords (e.g., "Getting Started with PostHog").
*   **Links to Key Resources:** Includes links to the docs, community, roadmap, and other important pages.
*   **Emphasis on Benefits:** Highlights the value of each feature.
*   **Clear Call to Actions:**  Encourages sign-ups and contributions.
*   **Updated Structure:** Improves readability.
*   **Alt Text:**  Added `alt` text to the images for accessibility and SEO.
*   **Keywords:** Added keyword terms like "open-source product analytics," "web analytics," "session replay," and other features.
*   **Self-hosting details:**  Included more details on the self-hosting.