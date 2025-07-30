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

## PostHog: The Open-Source Platform for Product Success

PostHog is your all-in-one, open-source solution for building better products by understanding and engaging your users.  ([View on GitHub](https://github.com/PostHog/posthog))

**Key Features:**

*   **Product Analytics:** Deep dive into user behavior with event-based analytics, visualizations, and SQL queries.
*   **Web Analytics:** Monitor website traffic, user sessions, conversions, and web vitals in a GA-like dashboard.
*   **Session Replays:**  Watch real user sessions to identify usability issues and understand user interactions.
*   **Feature Flags:**  Roll out new features safely and control access with feature flags.
*   **Experiments:**  Test different versions of your product and measure their impact on key metrics.
*   **Error Tracking:**  Monitor and resolve errors to improve product stability and user experience.
*   **Surveys:**  Gather user feedback with customizable surveys and templates.
*   **Data Warehouse & Pipelines:** Integrate data from external tools and transform incoming data for advanced analysis.
*   **LLM Observability:** Capture traces, generations, latency, and cost for your LLM-powered applications.

Enjoy all of these features for free with a generous monthly free tier; sign up for [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup) today.

## Table of Contents

-   [PostHog: The Open-Source Platform for Product Success](#posthog-the-open-source-platform-for-product-success)
-   [Table of Contents](#table-of-contents)
-   [Getting Started with PostHog](#getting-started-with-posthog)
    -   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    -   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
-   [Setting up PostHog](#setting-up-posthog)
-   [Learning More About PostHog](#learning-more-about-posthog)
-   [Contributing](#contributing)
-   [Open-source vs. paid](#open-source-vs-paid)
-   [We’re Hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

Get up and running quickly with PostHog by signing up for a free account on [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Enjoy a free tier that includes 1 million events, 5,000 recordings, 1 million flag requests, 100,000 exceptions, and 250 survey responses each month.

### Self-hosting the open-source hobby deploy (Advanced)

For self-hosting, deploy a hobby instance on Linux with Docker using this command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open-source deployments are recommended for up to 100k events per month.  For higher volumes, migrate to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud).

For more information on self-hosting, see the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer).

## Setting Up PostHog

Integrate PostHog into your product using the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), an [SDK](https://posthog.com/docs/getting-started/install?tab=sdks), or the [API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs are available for these languages:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Also, find documentation and guides for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

After installing PostHog, consult the [product docs](https://posthog.com/docs/product-os) to learn about product analytics, web analytics, session replays, feature flags, experiments, error tracking, surveys, data warehouse, and more.

## Learning More About PostHog

Explore our open-source [company handbook](https://posthog.com/handbook) to learn about our strategy, working methods, and processes.

For a guide on making the most of PostHog, check out [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled), which walks you through measuring activation, tracking retention, and capturing revenue.

## Contributing

We welcome your contributions!

*   Vote on features or get early access to beta functionality in our [roadmap](https://posthog.com/roadmap).
*   Submit a PR (see our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-source vs. paid

This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), with the exception of the `ee` directory.

For 100% Free and Open Source Software (FOSS), check out the [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

Pricing for paid plans can be found on [our pricing page](https://posthog.com/pricing).

## We’re Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

We're growing quickly and would love for you to join our team.  Find out more on our [careers page](https://posthog.com/careers).