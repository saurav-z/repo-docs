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

## **PostHog: The Open-Source Product Analytics Platform for Building Successful Products**

PostHog empowers product teams with the tools they need to understand user behavior, ship better products, and drive growth – all in one platform.  **(Click here to view the original repository: [PostHog on GitHub](https://github.com/PostHog/posthog))**

**Key Features:**

*   **Product Analytics:** Gain deep insights into user behavior with event-based analytics, custom dashboards, and SQL querying capabilities.
*   **Web Analytics:** Track website traffic, monitor key metrics like conversions and web vitals, and analyze user sessions.
*   **Session Replays:** Watch real user sessions to understand how users interact with your website or mobile app and troubleshoot issues.
*   **Feature Flags:** Safely release new features and A/B test changes on specific user segments.
*   **Experiments:** Conduct A/B tests to measure the impact of changes and optimize your product.
*   **Error Tracking:** Identify and resolve bugs quickly with real-time error monitoring and alerts.
*   **Surveys:** Gather user feedback with no-code survey templates and a powerful survey builder.
*   **Data Warehouse:** Integrate data from external sources like Stripe, HubSpot, and your data warehouse, and query it alongside your product data.
*   **Data Pipelines (CDP):** Transform and route your data to various tools and your data warehouse.
*   **LLM Observability:** Monitor your LLM-powered applications with tracing, generation tracking, and cost analysis.

PostHog offers a generous **free tier** to get you started. Sign up for [PostHog Cloud US](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup).

## Table of Contents

-   [PostHog: The Open-Source Product Analytics Platform for Building Successful Products](#posthog-the-open-source-product-analytics-platform-for-building-successful-products)
-   [Table of Contents](#table-of-contents)
-   [Getting Started with PostHog](#getting-started-with-posthog)
    -   [PostHog Cloud (Recommended)](#posthog-cloud-recommended)
    -   [Self-hosting the open-source hobby deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
-   [Setting up PostHog](#setting-up-posthog)
-   [Learning more about PostHog](#learning-more-about-posthog)
-   [Contributing](#contributing)
-   [Open-source vs. paid](#open-source-vs-paid)
-   [We’re hiring!](#were-hiring)

## Getting Started with PostHog

### PostHog Cloud (Recommended)

The fastest and easiest way to get started is by signing up for free to [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). The free tier includes 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses per month.

### Self-hosting the open-source hobby deploy (Advanced)

If you prefer self-hosting, deploy a hobby instance with Docker:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open source deployments are recommended for approximately 100k events per month. For larger volumes, we recommend migrating to [PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud).

_Note:_ Community support and guarantees are not provided for open source deployments. See the [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for further information.

## Setting up PostHog

Integrate PostHog by installing the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or by using our [API](https://posthog.com/docs/getting-started/install?tab=api).

We provide SDKs and libraries for these popular languages and frameworks:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Further resources available for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

For detailed setup instructions, explore our [product docs](https://posthog.com/docs/product-os) for insights on [product analytics](https://posthog.com/docs/product-analytics/capture-events), [web analytics](https://posthog.com/docs/web-analytics/getting-started), [session replays](https://posthog.com/docs/session-replay/how-to-watch-recordings), [feature flags](https://posthog.com/docs/feature-flags/creating-feature-flags), [experiments](https://posthog.com/docs/experiments/creating-an-experiment), [error tracking](https://posthog.com/docs/error-tracking/installation#setting-up-exception-autocapture), [surveys](https://posthog.com/docs/surveys/installation), [data warehouse](https://posthog.com/docs/cdp/sources), and more.

## Learning more about PostHog

Explore our open source [company handbook](https://posthog.com/handbook) to learn about our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

Discover how to maximize PostHog's potential with our guide to [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled) which includes instructions on [measuring activation](https://posthog.com/docs/new-to-posthog/activation), [tracking retention](https://posthog.com/docs/new-to-posthog/retention), and [capturing revenue](https://posthog.com/docs/new-to-posthog/revenue).

## Contributing

We welcome contributions of all sizes:

*   Suggest features and view our [roadmap](https://posthog.com/roadmap)
*   Submit a Pull Request: Review our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)
*   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)

## Open-source vs. paid

This repository is available under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory (which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE)).

For a 100% FOSS version, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

View the pricing for our paid plan on [our pricing page](https://posthog.com/pricing).

## We’re hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Thank you for reading!

Join our growing team: [PostHog Careers](https://posthog.com/careers)
```
Key improvements and SEO considerations:

*   **Clear, Concise Title:**  A strong heading that includes important keywords like "Open-Source," "Product Analytics," and the brand name.
*   **SEO-Optimized Introductory Sentence:** The hook immediately conveys the value proposition.
*   **Targeted Keywords:** Includes terms like "user behavior," "web analytics," "session replay," "feature flags," etc., which users might search for.
*   **Bulleted Key Features:** Clearly lists the platform's capabilities, improving readability and search engine understanding.
*   **Internal Linking:**  Uses links to crucial documentation pages like installation, SDKs, and product features.  This improves navigation and SEO.
*   **Call to Action (CTA):** Includes a clear call to action to sign up.
*   **Structured Content:** The Table of Contents and clear headings make the document easy to scan, improving user experience and SEO.
*   **Bolded Headings:** Helps emphasize important information, aiding readability.
*   **Emphasis on Open Source:**  Highlights the open-source nature, a key selling point.
*   **Concise Language:** Avoids overly complex phrasing, making the content accessible.
*   **Alt Text for Images:** Includes descriptive alt text for images, important for accessibility and SEO.
*   **Hiring section:** Includes a hiring section to encourage potential candidates to apply.
*   **Link back to original repo** Makes it easy for users to find the source code.