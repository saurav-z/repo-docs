<!-- Improved README for PostHog -->

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

PostHog is a powerful, open-source platform that gives you the tools to build a successful product by providing deep insights into user behavior and enabling data-driven decision-making.  Check out the original repo: [https://github.com/PostHog/posthog](https://github.com/PostHog/posthog)

**Key Features:**

*   **Product Analytics:** Understand user behavior with event-based analytics, analyze data, and visualize insights using dashboards or SQL.
*   **Web Analytics:** Monitor website traffic, track user sessions, conversion, web vitals, and revenue.
*   **Session Replays:** Watch real user sessions to understand how users interact with your website or app.
*   **Feature Flags:**  Roll out features to specific users or cohorts with feature flags.
*   **Experiments:** Test product changes and measure their impact on key metrics through A/B testing.
*   **Error Tracking:** Track and resolve errors to improve product stability and user experience.
*   **Surveys:** Gather user feedback using a no-code survey builder.
*   **Data Warehouse Integrations:** Sync data from external tools (Stripe, Hubspot, etc.) to centralize your data.
*   **Data Pipelines (CDP):** Run custom filters and transformations on your incoming data. Send it to 25+ tools or any webhook in real time or batch export large amounts to your warehouse.
*   **LLM Observability:** Capture traces, generations, latency, and cost for your LLM-powered applications.

**Free Tier:**

PostHog offers a generous free tier, so you can get started without any cost: [https://posthog.com/pricing](https://posthog.com/pricing).  Sign up for PostHog Cloud US [https://us.posthog.com/signup](https://us.posthog.com/signup) or PostHog Cloud EU [https://eu.posthog.com/signup](https://eu.posthog.com/signup).

## Getting Started

Choose your deployment method:

### PostHog Cloud (Recommended)

Get started quickly by signing up for a free account on [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

### Self-Hosting (Advanced)

For self-hosting, deploy a hobby instance using Docker:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

*Note: Self-hosting support is limited.  See the [self-hosting docs](https://posthog.com/docs/self-host) for more information.*

## Setting Up PostHog

Integrate PostHog with your project using a variety of SDKs and the PostHog API:

*   Install the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet)
*   Use one of the available [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks)
*   Integrate with the [API](https://posthog.com/docs/getting-started/install?tab=api)

**SDKs Available:**

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Find additional SDKs for Go, .NET/C#, Django, Angular, WordPress, and Webflow in the PostHog documentation.

**Product Documentation:** Explore PostHog's core capabilities:  [product docs](https://posthog.com/docs/product-os)

## Learn More

*   **Company Handbook:**  Learn about PostHog's [strategy](https://posthog.com/handbook/why-does-posthog-exist), [culture](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).
*   **Getting Started Guide:**  Discover how to use PostHog for [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

Help build PostHog:

*   Contribute to the [roadmap](https://posthog.com/roadmap)
*   Open a PR (See instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally))
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open Source vs. Paid

*   **MIT License:** This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)).
*   **100% FOSS?** Check out [posthog-foss](https://github.com/PostHog/posthog-foss)
*   **Pricing:**  View transparent pricing on [our pricing page](https://posthog.com/pricing).

## Join Our Team!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

We're hiring!  Join our growing team.  Apply here:  [https://posthog.com/careers](https://posthog.com/careers).