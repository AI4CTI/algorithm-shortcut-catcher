# Security policy

## Supported versions

`shortcut-catcher` is a research tool released as part of the publicly funded [AI4CTI](https://ai4cti.polito.it/) initiative. Only the **latest released minor version** receives security fixes.

| Version | Supported            |
| ------- | -------------------- |
| 0.1.x   | ✅ (current release) |
| < 0.1   | ❌                   |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive matters. Contact us privately at:

> **info@ermes.company**, with `[shortcut-catcher security]` in the subject line so the message reaches the maintainers.

Please include:

- a description of the issue and its impact;
- a minimal reproduction (command, configuration, synthetic data);
- the version you tested (`shortcut-catcher --version`).

## Process and timing

This is a research project, not a commercial product. We commit to:

- **acknowledgement** within **14 days**;
- an assessment and indicative remediation timeline within **30 days**;
- coordinated disclosure once a fix is available, crediting reporters unless they prefer to remain anonymous.

Vulnerabilities in third-party dependencies are forwarded to the upstream project.

## Out of scope

- Loading untrusted model files or configuration files from untrusted sources. Treat both as code.
- Issues that need write access to the machine running the tool.
