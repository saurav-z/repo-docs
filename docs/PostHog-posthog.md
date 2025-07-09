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

## PostHog: The Open-Source Product Analytics Platform for Growth

PostHog is an all-in-one, open-source product analytics platform empowering you to build successful products with data-driven insights.  [Explore the PostHog Repository](https://github.com/PostHog/posthog).

**Key Features:**

*   **Product Analytics:** Understand user behavior with autocapture or custom event tracking, analyze data through visualizations and SQL.
*   **Web Analytics:** Monitor website traffic, user sessions, conversion, web vitals, and revenue with a GA-like dashboard.
*   **Session Replays:** Watch user sessions to diagnose issues and understand user behavior.
*   **Feature Flags:** Safely roll out features to select users or cohorts.
*   **Experiments:** Test changes and measure their impact, including no-code experiment setup.
*   **Error Tracking:** Track and resolve errors to improve your product.
*   **Surveys:** Create and deploy no-code surveys to gather valuable user feedback.
*   **Data Warehouse & CDP:** Integrate with external data sources and transform your incoming data.
*   **LLM Observability:** Monitor traces, generations, latency, and cost for LLM-powered applications.

Benefit from a generous free tier and easily get started with [PostHog Cloud](https://us.posthog.com/signup) (US) or [PostHog Cloud EU](https://eu.posthog.com/signup).

## Table of Contents

-   [Getting Started with PostHog Cloud (Recommended)](#getting-started-with-posthog-cloud-recommended)
-   [Self-Hosting the Open-Source Hobby Deploy (Advanced)](#self-hosting-the-open-source-hobby-deploy-advanced)
-   [Setting Up PostHog](#setting-up-posthog)
-   [Learning More About PostHog](#learning-more-about-posthog)
-   [Contributing](#contributing)
-   [Open-source vs. Paid](#open-source-vs-paid)

## Getting Started with PostHog Cloud (Recommended)

Sign up for a free account with [PostHog Cloud](https://us.posthog.com/signup) or [PostHog Cloud EU](https://eu.posthog.com/signup). Enjoy a generous free tier including 1 million events, 5k recordings, 1M flag requests, 100k exceptions, and 250 survey responses per month.

## Self-Hosting the Open-Source Hobby Deploy (Advanced)

Self-host a hobby instance in one line on Linux with Docker (4GB memory recommended):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/posthog/posthog/HEAD/bin/deploy-hobby)"
```

Open source deployments are suitable for approximately 100k events per month.  We recommend [migrating to a PostHog Cloud](https://posthog.com/docs/migrate/migrate-to-cloud) after that.

We do not provide customer support for open-source deployments.  See our [self-hosting docs](https://posthog.com/docs/self-host), [troubleshooting guide](https://posthog.com/docs/self-host/deploy/troubleshooting), and [disclaimer](https://posthog.com/docs/self-host/open-source/disclaimer) for more information.

## Setting Up PostHog

Install PostHog using our [JavaScript web snippet](https://posthog.com/docs/getting-started/install?tab=snippet), [SDKs](https://posthog.com/docs/getting-started/install?tab=sdks), or [API](https://posthog.com/docs/getting-started/install?tab=api).

SDKs and libraries available for:

| Frontend                                              | Mobile                                                          | Backend                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| [JavaScript](https://posthog.com/docs/libraries/js)   | [React Native](https://posthog.com/docs/libraries/react-native) | [Python](https://posthog.com/docs/libraries/python) |
| [Next.js](https://posthog.com/docs/libraries/next-js) | [Android](https://posthog.com/docs/libraries/android)           | [Node](https://posthog.com/docs/libraries/node)     |
| [React](https://posthog.com/docs/libraries/react)     | [iOS](https://posthog.com/docs/libraries/ios)                   | [PHP](https://posthog.com/docs/libraries/php)       |
| [Vue](https://posthog.com/docs/libraries/vue-js)      | [Flutter](https://posthog.com/docs/libraries/flutter)           | [Ruby](https://posthog.com/docs/libraries/ruby)     |

Also, find docs and guides for [Go](https://posthog.com/docs/libraries/go), [.NET/C#](https://posthog.com/docs/libraries/dotnet), [Django](https://posthog.com/docs/libraries/django), [Angular](https://posthog.com/docs/libraries/angular), [WordPress](https://posthog.com/docs/libraries/wordpress), [Webflow](https://posthog.com/docs/libraries/webflow), and more.

After installing PostHog, consult the [product docs](https://posthog.com/docs/product-os) to learn more about product features.

## Learning More About PostHog

Explore our open-source [company handbook](https://posthog.com/handbook) for details on our [strategy](https://posthog.com/handbook/why-does-posthog-exist), [ways of working](https://posthog.com/handbook/company/culture), and [processes](https://posthog.com/handbook/team-structure).

Check out our guide to [winning with PostHog](https://posthog.com/docs/new-to-posthog/getting-hogpilled) for tips on measuring activation, tracking retention, and capturing revenue.

## Contributing

We welcome contributions of all sizes:

-   Vote on features or get early access to beta functionality in our [roadmap](https://posthog.com/roadmap)
-   Submit a [feature request](https://github.com/PostHog/posthog/issues/new?assignees=&labels=enhancement%2C+feature&template=feature_request.md) or [bug report](https://github.com/PostHog/posthog/issues/new?assignees=&labels=bug&template=bug_report.md)
-   Open a PR (see our instructions on [developing PostHog locally](https://posthog.com/handbook/engineering/developing-locally))

## Open-source vs. paid

This repository is licensed under the [MIT expat license](https://github.com/PostHog/posthog/blob/master/LICENSE), except for the `ee` directory, which has its [license here](https://github.com/PostHog/posthog/blob/master/ee/LICENSE).

For a fully FOSS version, check out our [posthog-foss](https://github.com/PostHog/posthog-foss) repository.

Pricing for paid plans is transparently available on [our pricing page](https://posthog.com/pricing).

## We’re hiring!

We're always looking for talented individuals. Check out our [careers page](https://posthog.com/careers) to join our team!

## Contributors 🦸

[//]: contributor-faces

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
 <a href="https://github.com/timgl"><img src="https://avatars.githubusercontent.com/u/1727427?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/mariusandra"><img src="https://avatars.githubusercontent.com/u/53387?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/EDsCODE"><img src="https://avatars.githubusercontent.com/u/13127476?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/Twixes"><img src="https://avatars.githubusercontent.com/u/4550621?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/macobo"><img src="https://avatars.githubusercontent.com/u/148820?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/paolodamico"><img src="https://avatars.githubusercontent.com/u/5864173?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/fuziontech"><img src="https://avatars.githubusercontent.com/u/391319?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/yakkomajuri"><img src="https://avatars.githubusercontent.com/u/38760734?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/jamesefhawkins"><img src="https://avatars.githubusercontent.com/u/47497682?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/posthog-bot"><img src="https://avatars.githubusercontent.com/u/69588470?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/apps/dependabot-preview"><img src="https://avatars.githubusercontent.com/in/2141?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/bhavish-agarwal"><img src="https://avatars.githubusercontent.com/u/14195048?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/Tannergoods"><img src="https://avatars.githubusercontent.com/u/60791437?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/ungless"><img src="https://avatars.githubusercontent.com/u/8397061?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/apps/dependabot"><img src="https://avatars.githubusercontent.com/in/29110?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/gzog"><img src="https://avatars.githubusercontent.com/u/1487006?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/samcaspus"><img src="https://avatars.githubusercontent.com/u/19220113?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/Tmunayyer"><img src="https://avatars.githubusercontent.com/u/29887304?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/adamb70"><img src="https://avatars.githubusercontent.com/u/11885987?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/SanketDG"><img src="https://avatars.githubusercontent.com/u/8980971?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/kpthatsme"><img src="https://avatars.githubusercontent.com/u/5965891?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/J0"><img src="https://avatars.githubusercontent.com/u/8011761?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/14MR"><img src="https://avatars.githubusercontent.com/u/5824170?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/03difoha"><img src="https://avatars.githubusercontent.com/u/8876615?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/ahtik"><img src="https://avatars.githubusercontent.com/u/140952?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/Algogator"><img src="https://avatars.githubusercontent.com/u/1433469?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/GalDayan"><img src="https://avatars.githubusercontent.com/u/24251369?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/Kacppian"><img src="https://avatars.githubusercontent.com/u/14990078?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/FUSAKLA"><img src="https://avatars.githubusercontent.com/u/6112562?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/iMerica"><img src="https://avatars.githubusercontent.com/u/487897?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/stevenphaedonos"><img src="https://avatars.githubusercontent.com/u/12955616?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/tapico-weyert"><img src="https://avatars.githubusercontent.com/u/70971917?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/adamschoenemann"><img src="https://avatars.githubusercontent.com/u/2095226?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/AlexandreBonaventure"><img src="https://avatars.githubusercontent.com/u/4596409?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/dan-dr"><img src="https://avatars.githubusercontent.com/u/6669808?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/dts"><img src="https://avatars.githubusercontent.com/u/273856?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/jamiehaywood"><img src="https://avatars.githubusercontent.com/u/26779712?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/rushabhnagda11"><img src="https://avatars.githubusercontent.com/u/3235568?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/weyert"><img src="https://avatars.githubusercontent.com/u/7049?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/casio"><img src="https://avatars.githubusercontent.com/u/29784?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/Hungsiro506"><img src="https://avatars.githubusercontent.com/u/10346923?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/bitbreakr"><img src="https://avatars.githubusercontent.com/u/3123986?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/edmorley"><img src="https://avatars.githubusercontent.com/u/501702?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/wundo"><img src="https://avatars.githubusercontent.com/u/113942?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/andreipopovici"><img src="https://avatars.githubusercontent.com/u/1143417?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/benjackwhite"><img src="https://avatars.githubusercontent.com/u/2536520?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/serhey-dev"><img src="https://avatars.githubusercontent.com/u/37838803?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/sjmadsen"><img src="https://avatars.githubusercontent.com/u/57522?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/piemets"><img src="https://avatars.githubusercontent.com/u/70321811?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/eltjehelene"><img src="https://avatars.githubusercontent.com/u/75622766?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/athreyaanand"><img src="https://avatars.githubusercontent.com/u/31478366?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/berntgl"><img src="https://avatars.githubusercontent.com/u/55957336?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/fakela"><img src="https://avatars.githubusercontent.com/u/39309699?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/seanpackham"><img src="https://avatars.githubusercontent.com/u/3830791?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/corywatilo"><img src="https://avatars.githubusercontent.com/u/154479?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/mikenicklas"><img src="https://avatars.githubusercontent.com/u/6363580?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/lottiecoxon"><img src="https://avatars.githubusercontent.com/u/65415371?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/oshura3"><img src="https://avatars.githubusercontent.com/u/30472479?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/Abo7atm"><img src="https://avatars.githubusercontent.com/u/33042538?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/brianetaveras"><img src="https://avatars.githubusercontent.com/u/52111440?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/callumgare"><img src="https://avatars.githubusercontent.com/u/346340?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/RedFrez"><img src="https://avatars.githubusercontent.com/u/30352852?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/cirdes"><img src="https://avatars.githubusercontent.com/u/727781?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/DannyBen"><img src="https://avatars.githubusercontent.com/u/2405099?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/sj26"><img src="https://avatars.githubusercontent.com/u/14028?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/paulanunda"><img src="https://avatars.githubusercontent.com/u/155981?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/arosales"><img src="https://avatars.githubusercontent.com/u/1707853?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/ChandanSagar"><img src="https://avatars.githubusercontent.com/u/27363164?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/wadenick"><img src="https://avatars.githubusercontent.com/u/9014043?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/jgannondo"><img src="https://avatars.githubusercontent.com/u/28159071?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/keladhruv"><img src="https://avatars.githubusercontent.com/u/30433468?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/grellyd"><img src="https://avatars.githubusercontent.com/u/7812612?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/rberrelleza"><img src="https://avatars.githubusercontent.com/u/475313?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/annanay25"><img src="https://avatars.githubusercontent.com/u/10982987?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/cohix"><img src="https://avatars.githubusercontent.com/u/5942370?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/gouthamve"><img src="https://avatars.githubusercontent.com/u/7354143?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/alexellis"><img src="https://avatars.githubusercontent.com/u/6358735?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/prologic"><img src="https://avatars.githubusercontent.com/u/1290234?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/jgustie"><img src="https://avatars.githubusercontent.com/u/883981?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/kubemq"><img src="https://avatars.githubusercontent.com/u/45835100?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/vania-pooh"><img src="https://avatars.githubusercontent.com/u/829320?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/irespaldiza"><img src="https://avatars.githubusercontent.com/u/11633327?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/croomes"><img src="https://avatars.githubusercontent.com/u/211994?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/snormore"><img src="https://avatars.githubusercontent.com/u/182290?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/faik"><img src="https://avatars.githubusercontent.com/u/43129?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/aandryashin"><img src="https://avatars.githubusercontent.com/u/1412461?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/andrewsomething"><img src="https://avatars.githubusercontent.com/u/46943?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/Ferroin"><img src="https://avatars.githubusercontent.com/u/905151?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/cpanato"><img src="https://avatars.githubusercontent.com/u/4115580?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/cakrit"><img src="https://avatars.githubusercontent.com/u/43294513?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/dkhenry"><img src="https://avatars.githubusercontent.com/u/489643?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/oxplot"><img src="https://avatars.githubusercontent.com/u/483682?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/marc-barry"><img src="https://avatars.githubusercontent.com/u/4965634?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/moabu"><img src="https://avatars.githubusercontent.com/u/47318409?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/nawazdhandala"><img src="https://avatars.githubusercontent.com/u/2697338?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/dar-mehta"><img src="https://avatars.githubusercontent.com/u/10489943?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/gmmorris"><img src="https://avatars.githubusercontent.com/u/386208?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/bitdeli-chef"><img src="https://avatars.githubusercontent.com/u/3092978?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/nsidartha"><img src="https://avatars.githubusercontent.com/u/26918226?v=4" width="50" height="50" alt=""/></a> <a href="http://massimilianomirra.com/"><img src="https://avatars.githubusercontent.com/u/19322?v=4" width="50" height="50" alt=""/></a> <a href="https://www.bronsonavila.com/"><img src="https://avatars.githubusercontent.com/u/30540995?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/apps/posthog-contributions-bot"><img src="https://avatars.githubusercontent.com/in/105985?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/joesaunderson"><img src="https://avatars.githubusercontent.com/u/11272509?v=4" width="50" height="50" alt=""/></a> <a href="https://www.ianlai.dev/"><img src="https://avatars.githubusercontent.com/u/68859?v=4" width="50" height="50" alt=""/></a> <a href="http://martinmck.com"><img src="https://avatars.githubusercontent.com/u/11256663?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/lharress"><img src="https://avatars.githubusercontent.com/u/13482930?v=4" width="50" height="50" alt=""/></a> <a href="https://www.linkedin.com/in/adrien-brault-4b987426/"><img src="https://avatars.githubusercontent.com/u/611271?v=4" width="50" height="50" alt=""/></a> <a href="https://leggetter.co.uk"><img src="https://avatars.githubusercontent.com/u/328367?v=4" width="50" height="50" alt=""/></a> <a href="https://github.com/wushaobo"><img src="