# Security Policy

## Supported versions

The project is pre-1.0. Security fixes are made against the latest `0.x`
release and the `main` branch.

## Reporting a vulnerability

Please report suspected vulnerabilities privately, not in a public issue.

- Preferred: open a [private security advisory](https://github.com/supportlibrariesp4droid/residue-rs/security/advisories/new)
  on GitHub.
- Alternatively, email **supportlibrariesp4@gmail.com** with the details.

Include a description, a reproduction if you have one, and the version or commit
affected. You can expect an acknowledgement within a few days.

## Scope

This is a pure-Rust library with `#![forbid(unsafe_code)]`, so memory-safety
bugs of the usual kind cannot originate in this crate. The security-relevant
failure mode here is a **correctness bug that produces a silently wrong result**
— for example an operation that should detect overflow but does not, or a value
that compares or reconstructs incorrectly. Reports of that kind are treated with
the same seriousness as a memory-safety issue and are very welcome.
