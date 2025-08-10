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

## PostHog: The Open-Source Product Analytics Platform for Growth

PostHog is an all-in-one, open-source platform designed to help you build successful products by providing powerful analytics and user behavior insights. Explore the [PostHog GitHub Repository](https://github.com/PostHog/posthog).

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, visualization, and SQL queries.
*   **Web Analytics:** Monitor web traffic, user sessions, conversion, web vitals, and revenue.
*   **Session Replays:** Watch real user sessions to diagnose issues and improve user experience.
*   **Feature Flags:** Safely roll out features to specific users or cohorts.
*   **Experiments:** Test changes and measure their impact using A/B testing.
*   **Error Tracking:** Identify and resolve issues with error tracking and alerts.
*   **Surveys:** Gather feedback with no-code survey templates and a builder.
*   **Data Warehouse:** Integrate data from external tools for comprehensive analysis.
*   **Data Pipelines (CDP):** Transform and route your data to 25+ tools.
*   **LLM Observability:** Monitor traces, generations, latency, and cost for your LLM-powered app.

PostHog offers a generous [free tier](https://posthog.com/pricing) and is available as [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

## Table of Contents

*   [PostHog: The Open-Source Product Analytics Platform for Growth](#posthog-the-open-source-product-analytics-platform-for-growth)
*   [Key Features](#key-features)
*   [Getting Started with PostHog](#getting-started-with-posthog)
    *   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    *   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
*   [Setting up PostHog](#setting-up-posthog)
*   [Learning more about PostHog](#learning-more-about-posthog)
*   [Contributing](#contributing)
*   [Open-source vs. paid](#open-source-vs-paid)
*   [We’re hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

Get started quickly and reliably by signing up for [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). The free tier includes 1 million events, 5,000 recordings, 1M flag requests, 100k exceptions, and 250 survey responses monthly.

### Self-hosting the open-source hobby deploy (Advanced)

Self-host PostHog with a single Docker command (requires 4GB memory):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open source deployments are suitable for up to 100k events monthly. For higher usage, migrate to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud).

Note:  Customer support and guarantees aren't provided for open-source deployments.  Refer to the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer).

## Setting up PostHog

Integrate PostHog by using the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), an [SDK](https://posthog.com/docs/getting-started/install?tab=sdks), or the [API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs and libraries are available for various languages and frameworks, including:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Additional documentation and guides cover Go, .NET/C#, Django, Angular, WordPress, and Webflow. For details on setting up features like product analytics, web analytics, session replays, feature flags, experiments, error tracking, surveys, and data warehouse, explore the [product docs](https://posthog.com/docs/product-os).

## Learning more about PostHog

Explore open-source resources like our [company handbook](https://posthog.com/handbook) for insights into PostHog's [strategy](https://posthog.com/handbook/why-does-posthog-exist), [culture](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

Discover tips to maximize PostHog's features with our guide on [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled), which covers measuring activation, tracking retention, and capturing revenue.

## Contributing

Contribute and help shape PostHog:

*   Suggest features or participate in beta programs via our [roadmap](https://posthog.com/roadmap).
*   Submit a PR. See instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or a [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-source vs. paid

This repository uses the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory. The `ee` directory's license is [here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE).

For a fully open-source solution, review our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

Transparent pricing information for paid plans is available on [our pricing page](https://posthog.com/pricing).

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

We're expanding our team. Check out our [careers page](https://posthog.com/careers) to explore opportunities.
```

**Key improvements and explanations:**

*   **SEO-optimized title and introduction:**  Used "PostHog" and "Open-Source Product Analytics" in the title and first sentence to target relevant search terms.
*   **Concise Hook:** The first sentence directly highlights the value proposition: helping users build successful products.
*   **Clear Headings:** Used proper markdown headings for improved readability and SEO.
*   **Bulleted Key Features:** Lists key features in a clear, scannable bulleted format.  Added more details to each feature to increase keyword relevance.
*   **Internal Linking:** Links to relevant sections within the README for user navigation.
*   **Call to Action (CTA):** The "We're hiring" section includes a clear call to action.
*   **Keywords:** Incorporated relevant keywords throughout the README, such as "product analytics," "web analytics," "session replay," "feature flags," etc.
*   **Removed Redundancy:**  Consolidated information and removed unnecessary repetition.
*   **Concise Sections:**  Simplified and shortened some sections for better readability.
*   **Formatting:** Improved the overall formatting for visual appeal and readability.
*   **Actionable instructions:** The section on getting started is written so users can quickly begin using the software.
*   **Emphasis on Open Source:**  Clearly states the open-source nature of the project, attracting those specifically looking for this.