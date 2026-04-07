## Deploying the Personal Academic Website

### Option A: GitHub Pages (Recommended)

1. Create a new repository: `ishrith-gowda/ishrith-gowda.github.io`

2. Copy this entire `website/` directory as the repo root:
   ```bash
   cp -r docs/website/* /path/to/ishrith-gowda.github.io/
   cd /path/to/ishrith-gowda.github.io
   git init && git add -A && git commit -m "initial academic website"
   git remote add origin git@github.com:ishrith-gowda/ishrith-gowda.github.io.git
   git push -u origin main
   ```

3. Add a GitHub Actions workflow at `.github/workflows/deploy.yml`:
   ```yaml
   name: deploy hugo site
   on:
     push:
       branches: [main]
   jobs:
     build:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
           with:
             fetch-depth: 0
         - name: setup hugo
           uses: peaceiris/actions-hugo@v3
           with:
             hugo-version: 'latest'
             extended: true
         - name: setup go
           uses: actions/setup-go@v5
           with:
             go-version: '1.21'
         - name: build
           run: hugo --minify
         - name: deploy
           uses: peaceiris/actions-gh-pages@v4
           with:
             github_token: ${{ secrets.GITHUB_TOKEN }}
             publish_dir: ./public
   ```

4. In repo Settings > Pages, set source to `gh-pages` branch.

5. Site will be live at `https://ishrith-gowda.github.io`

### Option B: Netlify

1. Push to any GitHub repo
2. Connect repo to Netlify
3. Build command: `hugo --minify`
4. Publish directory: `public`

### Customization

- **Photo**: Add `avatar.jpg` to `content/authors/admin/`
- **CV**: Add `cv.pdf` to `static/uploads/`
- **Custom domain**: Add CNAME file to `static/` with your domain
- **Additional publications**: Create new directories under `content/publication/`
