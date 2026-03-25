# Comprehensive Researcher Profile Setup Guide

**Ishrith Gowda | UC Berkeley EECS | Medical AI / Brain MRI Harmonization**

This document provides step-by-step instructions for setting up a world-class researcher profile across all major academic platforms. Follow the priority order below.

---

## Current Paper

- **Title:** SA-CycleGAN-2.5D: Self-Attention CycleGAN with Tri-Planar Context for Multi-Site MRI Harmonization
- **arXiv:** [2603.17219](https://arxiv.org/abs/2603.17219)
- **DOI:** 10.48550/arXiv.2603.17219
- **Categories:** cs.CV (primary), cs.AI, cs.LG
- **Authors:** Ishrith Gowda, Chunwei Liu
- **Status:** Preprint. Under review at MICCAI 2026

---

## Immediate arXiv Actions

### 1. Cross-List to eess.IV

Your paper is currently listed under cs.CV, cs.AI, cs.LG. You should **add eess.IV** (Image and Video Processing) as a cross-list. This is the primary home for medical imaging papers and will significantly increase visibility among the MRI/medical imaging community.

**How:** Go to https://arxiv.org/user > find paper 2603.17219 > click "Cross list" > add `eess.IV`

### 2. Link Code & Data

**How:** On the arXiv user page, click the "Link code & data" icon (the barcode icon) next to your paper. This redirects to Papers with Code where you link:
- **Official Code:** `https://github.com/ishrith-gowda/NeuroScope`
- Mark as "Official Implementation"
- Add tasks: "MRI Harmonization", "Domain Adaptation", "Image-to-Image Translation"
- Add datasets: "BraTS-TCGA-GBM", "UPenn-GBM"

### 3. Update Comments Field (Optional)

Consider updating the comments field to include the code URL:
```
12 pages, 5 figures, 5 tables. Submitted to MICCAI 2026. Code: https://github.com/ishrith-gowda/NeuroScope
```

---

## Priority 1: Foundational Identifiers

### ORCID (https://orcid.org/register)

ORCID is a persistent digital identifier that uniquely distinguishes you. Required by NIH, most publishers, and funders. This is the universal connector between all platforms.

1. Register with `ishrithgowda@berkeley.edu`
2. Add a personal email as backup (you lose institutional email when you leave)
3. Complete all sections:
   - **Names:** Ishrith Gowda
   - **Affiliation:** University of California, Berkeley; Department of EECS
   - **Education:** UC Berkeley (and any prior institutions)
   - **Works:** Import arXiv paper 2603.17219 via DOI (10.48550/arXiv.2603.17219)
   - **Keywords:** medical image analysis, brain MRI harmonization, deep learning, domain adaptation, generative adversarial networks
   - **Websites:** GitHub, Google Scholar, personal website (once created)
4. Set all visibility to "Everyone"
5. Verify Berkeley email
6. **Save your ORCID iD** -- you will add it to every other platform

> After creating ORCID, update `CITATION.cff` in the repo with your ORCID iD.

### Google Scholar (https://scholar.google.com/citations)

The most widely used academic search engine. First place researchers check someone's work.

1. Sign in with Google account (use or link Berkeley email)
2. Name: "Ishrith Gowda"
3. Affiliation: "University of California, Berkeley"
4. Email: `ishrithgowda@berkeley.edu` (for verification checkmark)
5. Keywords (up to 5): "Medical Image Analysis", "Brain MRI", "Domain Adaptation", "Deep Learning", "Harmonization"
6. Confirm your arXiv paper when suggested
7. Add professional headshot
8. **Make profile public**
9. Enable automatic updates for new papers
10. Save your Google Scholar profile URL

### OpenReview (https://openreview.net/signup)

Required for submitting to NeurIPS, ICML, ICLR, MICCAI, and most top ML/AI venues.

1. Register with `ishrithgowda@berkeley.edu` (institutional emails activate immediately)
2. Complete all sections:
   - **Names:** Ishrith Gowda
   - **Emails:** Berkeley email + personal email
   - **Personal Links:** DBLP URL, Google Scholar URL, Semantic Scholar URL, LinkedIn, homepage
   - **History:** UC Berkeley, EECS, Student, start date
   - **Relations:** Add advisor(s) and collaborator (Chunwei Liu, Purdue)
   - **Expertise:** medical image analysis, MRI harmonization, deep learning, GANs, domain adaptation, federated learning
3. Import papers from Semantic Scholar if available
4. Link ORCID

---

## Priority 2: Discovery & Visibility Platforms

### Semantic Scholar (https://www.semanticscholar.org)

AI-powered academic search engine by Allen Institute for AI. Tracks influential citations and generates TLDRs.

1. Search for your paper or name on semanticscholar.org
2. Find your auto-generated author page
3. Click "Claim Author Page"
4. Verify with email, name, affiliation, ORCID
5. Update: displayed name, UC Berkeley affiliation, ORCID link
6. Remove any incorrectly attributed papers
7. Save your Semantic Scholar profile URL

### Papers with Code (https://paperswithcode.com)

Links papers to code, datasets, and benchmarks. Integrated directly into arXiv abstract pages.

1. Search for your paper on paperswithcode.com (it should auto-appear from arXiv within 24-48 hours)
2. If not auto-detected, manually add the paper
3. Click "Add Code" and link `https://github.com/ishrith-gowda/NeuroScope`
4. Mark as "Official Implementation"
5. Add relevant tasks:
   - "MRI Harmonization" (create this task if it doesn't exist)
   - "Domain Adaptation"
   - "Image-to-Image Translation"
6. Add datasets: BraTS, UPenn-GBM
7. Add results/benchmarks from your paper's tables

### DBLP (https://dblp.org)

Computer science bibliography. Auto-indexes from arXiv.

- DBLP should auto-index your paper from arXiv (may take 1-2 weeks)
- Check https://dblp.org/search?q=ishrith+gowda periodically
- Once indexed, your DBLP page URL is used by OpenReview and other platforms
- No action needed beyond verifying it appears correctly

### GitHub Profile Optimization

1. **Create a profile README** (create repo `ishrith-gowda/ishrith-gowda` with README.md):

```markdown
### Ishrith Gowda

UC Berkeley EECS | Medical AI Researcher

Currently working on brain MRI harmonization using deep learning -- developing methods to standardize multi-institutional neuroimaging data for improved clinical analysis.

**Publications:**
- [SA-CycleGAN-2.5D](https://arxiv.org/abs/2603.17219) -- Self-Attention CycleGAN with Tri-Planar Context for Multi-Site MRI Harmonization (MICCAI 2026, under review)

**Links:** [Google Scholar](YOUR_SCHOLAR_URL) | [ORCID](https://orcid.org/YOUR_ORCID) | [Semantic Scholar](YOUR_S2_URL)
```

2. **Pin the NeuroScope repo** to your GitHub profile
3. **Profile settings:**
   - Professional photo (same across all platforms)
   - Bio: "UC Berkeley EECS | Medical AI Researcher"
   - Location: Berkeley, CA
   - Link to personal website or arXiv page

---

## Priority 3: Professional Presence

### Personal Academic Website

Your canonical online presence. Recommended: Hugo + GitHub Pages with an academic theme.

**Quick setup with GitHub Pages:**
1. Create repo `ishrith-gowda.github.io`
2. Use Hugo Academic theme (now HugoBlox) or Jekyll Academic Pages
3. Sections to include:
   - **Home:** Photo, name, one-paragraph bio, affiliation, research interests, links
   - **Publications:** Full citation with PDF, arXiv, code, BibTeX links
   - **Research:** Plain-language project descriptions with figures
   - **CV:** Downloadable PDF
4. Add custom domain if desired (e.g., ishrithgowda.com)
5. Link this website from all other profiles

### Academic Social Media (Bluesky / X)

For paper promotion and community building:
- **Bluesky** (recommended primary): Growing academic community, less noise
- **X/Twitter** (secondary): Still has reach but declining academic engagement
- Post paper announcements with: title, 1-sentence summary, key figure, arXiv link, GitHub link
- Follow and engage with researchers in medical imaging, MICCAI community

### LinkedIn

Update your LinkedIn with:
- Headline: "UC Berkeley EECS | Medical AI Researcher"
- About section: Research focus and interests
- Publications section: Add your arXiv paper
- Featured section: Pin your paper and GitHub repo

### ResearchGate (https://www.researchgate.net)

Optional but can help with discoverability:
- Auto-indexes from arXiv
- Good for connecting with medical imaging researchers
- Upload preprint PDF
- Lower priority than the platforms above

---

## Priority 4: Future Platforms (Become Relevant After Publication)

### Web of Science / Scopus Author ID

These become relevant once your paper is published in a peer-reviewed journal or conference proceedings:
- Web of Science: Author ID created when your published paper is indexed
- Scopus: Author ID auto-generated when conference proceedings (e.g., MICCAI via Springer LNCS) are indexed
- After MICCAI acceptance, your Scopus Author ID will be created automatically
- Link these to ORCID when available

---

## Cross-Integration Checklist

Once all profiles are created, cross-link everything:

| Platform | Links To |
|---|---|
| ORCID | Google Scholar, Semantic Scholar, GitHub, personal website, all papers via DOI |
| Google Scholar | Personal website (in profile settings) |
| OpenReview | DBLP, Google Scholar, Semantic Scholar, LinkedIn, personal website, ORCID |
| Semantic Scholar | ORCID |
| GitHub | arXiv (repo homepage URL), personal website |
| Personal Website | All of the above |
| arXiv | Code link via Papers with Code |
| Papers with Code | GitHub repo |
| LinkedIn | Personal website, Google Scholar |

---

## Consistency Checklist

Use the EXACT same information everywhere:

- **Name:** Ishrith Gowda
- **Affiliation:** University of California, Berkeley
- **Department:** Department of Electrical Engineering and Computer Sciences (EECS)
- **Email:** ishrithgowda@berkeley.edu
- **Photo:** Same professional headshot on all platforms
- **Bio:** Consistent one-liner research description
- **ORCID:** Same iD linked everywhere

---

## Recommended Order of Operations

1. ORCID (5 min) -- foundation for everything
2. Google Scholar (5 min) -- most important visibility
3. OpenReview (10 min) -- needed for conference submissions
4. Semantic Scholar claim (5 min) -- auto-generated, just claim it
5. Papers with Code link (5 min) -- via arXiv "Link code & data"
6. arXiv cross-list to eess.IV (2 min) -- one click
7. GitHub profile README (15 min) -- public-facing code presence
8. Personal website (1-2 hours) -- can be done later but high value
9. LinkedIn update (10 min)
10. Academic social media (ongoing)

**Total estimated time for critical platforms (1-7): ~45 minutes**

---

## BibTeX Citation Block

Use this consistently across all platforms:

```bibtex
@article{gowda2026sacyclegan25d,
  title={SA-CycleGAN-2.5D: Self-Attention CycleGAN with Tri-Planar Context for Multi-Site MRI Harmonization},
  author={Gowda, Ishrith and Liu, Chunwei},
  journal={arXiv preprint arXiv:2603.17219},
  year={2026},
  url={https://arxiv.org/abs/2603.17219},
  doi={10.48550/arXiv.2603.17219}
}
```

> Update this BibTeX block everywhere once the paper is accepted at a venue.
