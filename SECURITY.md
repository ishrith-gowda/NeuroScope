# security policy

## supported versions

this is a research codebase released for reproducibility. security fixes are applied to the latest `main` only.

| version | supported |
|---------|-----------|
| latest `main` | yes |
| tagged releases | best effort |

## reporting a vulnerability

please do **not** open a public issue for security-sensitive reports.

instead, use github's private vulnerability reporting: open the repository's **security** tab and click **report a vulnerability**, or email **ishrithgowda@berkeley.edu** with the details.

please include:
- a description of the issue and its impact,
- steps to reproduce (or a proof of concept),
- any suggested remediation.

we aim to acknowledge reports within 7 days.

## scope and notes

- model checkpoints are loaded with `torch.load`. only load checkpoints from trusted sources; see the `weights_only` handling in the loaders.
- the repository runs `bandit` in ci as part of its security scanning.
