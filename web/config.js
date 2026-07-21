// Backend URL for GitHub Pages → Cloud Run split deployment.
// This file is overwritten by GitHub Actions during `Deploy to GitHub Pages`,
// and by the Docker build for all-in-one deployments (see Dockerfile).
// Empty string = same-origin (all-in-one deployment on Railway / Cloud Run).
window.__BACKEND_URL__ = '';
window.__FRONTEND_VERSION__ = 'dev';
