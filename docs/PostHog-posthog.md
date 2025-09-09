<p align="center">
  <img alt="PostHog Logo" src="https://user-images.githubusercontent.com/65415371/205059737-c8a4f836-4889-4654-902e-f302b160a.png">
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

PostHog is your all-in-one, open-source solution for building successful products, providing a comprehensive suite of tools to understand, analyze, and improve user behavior.  ([View the original repository](https://github.com/PostHog/posthog))

**Key Features:**

*   **Product Analytics:**  Track events, analyze user behavior, and visualize data with powerful tools including SQL support.
*   **Web Analytics:** Monitor website traffic, user sessions, conversion rates, and web vitals with a GA-like dashboard.
*   **Session Replays:** Watch recordings of real user sessions to understand user interactions and diagnose issues.
*   **Feature Flags:**  Safely roll out new features to specific user segments with feature flags.
*   **Experiments:**  Test changes and measure their impact on key metrics using A/B testing.
*   **Error Tracking:** Identify, track, and resolve errors to improve product stability.
*   **Surveys:** Gather user feedback with customizable surveys and pre-built templates.
*   **Data Warehouse:**  Integrate data from external sources like Stripe and Hubspot for unified analysis.
*   **Data Pipelines:** Transform and route data in real-time to various destinations.
*   **LLM Analytics:**  Track traces, generations, latency, and cost for LLM-powered applications.

Get started with PostHog Cloud (US or EU) for a generous free tier or self-host for maximum control!

## Table of Contents

-   [PostHog: The Open-Source Platform for Product Success](#posthog-the-open-source-platform-for-product-success)
-   [Table of Contents](#table-of-contents)
-   [Getting Started with PostHog](#getting-started-with-posthog)
    -   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    -   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
-   [Setting up PostHog](#setting-up-posthog)
-   [Learning More about PostHog](#learning-more-about-posthog)
-   [Contributing](#contributing)
-   [Open-source vs. paid](#open-source-vs-paid)
-   [We’re hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The quickest and most reliable way to get started is by signing up for a free [PostHog Cloud](https://us.posthog.com/signup) account (or [EU](https://eu.posthog.com/signup)). Benefit from a generous free tier that covers your first 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 1500 survey responses monthly.

### Self-hosting the open-source hobby deploy (Advanced)

For self-hosting, deploy a hobby instance using the following Docker command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open source deployments are suitable for up to approximately 100k events per month. For increased scale, we recommend migrating to PostHog Cloud.

We do not offer customer support or guarantees for self-hosted open-source deployments. Refer to our [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more information.

## Setting up PostHog

Set up your PostHog instance by integrating our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or by using our [API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs and libraries are available for numerous languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Additional documentation and guides are available for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

Once installed, explore the [product docs](https://posthog.com/docs/product-os) for detailed information on [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning More about PostHog

Our commitment to openness extends beyond code. Review our [company handbook](https://posthog.com/handbook), which details our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

For tips on maximizing PostHog's potential, see our guide, "[Winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled)," which covers [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

Contributions of all sizes are welcome:

*   Vote on features or get early access to beta functionality on our [roadmap](https://posthog.com/roadmap).
*   Submit a pull request (see our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)).
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

## Open-source vs. paid

This repository is licensed under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (licensed [here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)), if applicable.

For a completely FOSS experience, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository, which is devoid of proprietary code and features.

Transparent pricing information for our paid plans is available on [our pricing page](https://posthog.com/pricing).

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

If you made it this far, consider joining our team! We're expanding and [would love for you to join us](https://posthog.com/careers).
```
Key improvements and SEO considerations:

*   **Strong Headline:**  Clear, concise headline with a keyphrase ("Open-Source Platform for Product Success").
*   **Concise Summary:** A strong first sentence that highlights the core value proposition.
*   **Keyword Optimization:** Used keywords naturally ("product analytics," "web analytics," "session replays," "feature flags," "open source").
*   **Bullet Points:**  Easy-to-scan bulleted lists for features.
*   **Internal Linking:**  Links to core features within the document.
*   **Call to Action:** Encourages users to get started.
*   **Structure:** Table of Contents for easy navigation.
*   **Clean Formatting:** Improved readability with consistent headings and spacing.
*   **Emphasis on Value:** Focuses on the benefits users gain from using PostHog.
*   **"Why PostHog" is explained within the description**  Instead of just a link.
*   **Link Back:** Added a link back to the original repository.
*   **Clear Value Proposition:** The intro now immediately presents the core value to users.