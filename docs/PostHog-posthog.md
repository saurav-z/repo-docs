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

## PostHog: The Open-Source Platform for Product-Led Growth

**PostHog is a comprehensive, open-source platform designed to help you build better products and understand your users like never before.** (Check out the original repo: [PostHog on GitHub](https://github.com/PostHog/posthog))

### Key Features:

*   **Product Analytics:** Gain deep insights into user behavior with event-based analytics, including autocapture and custom instrumentation. Visualize data with built-in dashboards or SQL.
*   **Web Analytics:** Track website traffic, monitor user sessions, and analyze key metrics like conversion, web vitals, and revenue with a GA-like dashboard.
*   **Session Replays:**  Watch real user sessions to understand user interactions, diagnose issues, and improve your product.
*   **Feature Flags:**  Control feature releases and A/B test with feature flags, empowering you to safely roll out features and experiment with new ideas.
*   **Experiments:**  Run A/B tests to measure the impact of changes on key metrics. Utilize no-code experimentation for easy setup.
*   **Error Tracking:** Track errors and exceptions, receive alerts, and resolve issues to improve product stability.
*   **Surveys:**  Collect user feedback using no-code survey templates or build custom surveys to understand your users.
*   **Data Warehouse & Pipelines (CDP):** Integrate data from external sources (Stripe, Hubspot, etc.) for comprehensive analysis. Transform and route data to 25+ tools or your data warehouse.
*   **LLM Observability:**  Monitor traces, generations, latency, and cost for your LLM-powered applications.

### Getting Started:

*   **PostHog Cloud (Recommended):** The easiest way to start is with our free PostHog Cloud! Sign up at [US](https://us.posthog.com/signup) or [EU](https://eu.posthog.com/signup).  Enjoy a generous free tier.
*   **Self-Hosting (Advanced):** Deploy a hobby instance using Docker with the one-line command:
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
    ```
    *Note: Open source deployments are not supported. See [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer).*

### Setting Up PostHog:

1.  **Install:** Integrate PostHog with your product using the [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), one of our [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or our [API](https://posthog.com/docs/getting-started/install?tab=api).
2.  **SDKs & Libraries:** We offer SDKs for popular languages and frameworks:
    *   JavaScript, Next.js, React, Vue
    *   React Native, Android, iOS, Flutter
    *   Python, Node, PHP, Ruby
    *   Go, .NET/C#, Django, Angular, WordPress, Webflow
3.  **Explore:** Find setup guides in our [product docs](https://posthog.com/docs/product-os).

### Learn More:

*   **Company Handbook:** Explore our open-source [company handbook](https://posthog.com/handbook) to learn more about our strategy, culture, and processes.
*   **Winning with PostHog:** Discover how to measure activation, track retention, and capture revenue with our guide: [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled).

### Contribute:

We welcome contributions of all sizes!

*   Vote on features/get early access in our [roadmap](https://posthog.com/roadmap).
*   Submit a pull request (see instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally)).
*   Suggest [feature requests](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug reports](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md).

### Open-Source vs. Paid:

*   This repository is MIT licensed (except for the `ee` directory).
*   For a 100% FOSS version, check out [posthog-foss](https://github.com/PostHog/posthog-foss).
*   See our transparent [pricing page](https://posthog.com/pricing) for paid plans.

### We're Hiring!

<img src="https://res.cloudinary.com/dmukukwp6/image/upload/v1/posthog.com/src/components/Home/images/mission-control-hog" alt="Hedgehog working on a Mission Control Center" width="350px"/>

Want to join our fast-growing team?  Check out our [careers page](https://posthog.com/careers)!