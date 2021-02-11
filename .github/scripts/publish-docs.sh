cd docs

make html

git status
git add _build || true

git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git config user.name "github-actions[bot]"

# If there aren't changes, don't make a commit; push is no-op
git commit -m "Generate Python docs from $GITHUB_SHA" || true
git status
git push
