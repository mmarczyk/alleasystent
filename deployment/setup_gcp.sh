#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# AllEasystent — GCP Infrastructure Setup Script
#
# Run once to provision all required GCP resources.
# Prerequisites: gcloud CLI authenticated, PROJECT_ID set.
# Usage: PROJECT_ID=my-project bash deployment/setup_gcp.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?PROJECT_ID env var must be set}"
REGION="${REGION:-europe-central2}"
SA_NAME="alleasystent-sa"
REPO_NAME="alleasystent"

echo "▶ Setting project: $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

# ── Enable APIs ───────────────────────────────────────────────────────────────
echo "▶ Enabling required APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  pubsub.googleapis.com \
  secretmanager.googleapis.com \
  cloudscheduler.googleapis.com \
  --project="$PROJECT_ID"

# ── Artifact Registry ─────────────────────────────────────────────────────────
echo "▶ Creating Artifact Registry repository..."
gcloud artifacts repositories create "$REPO_NAME" \
  --repository-format=docker \
  --location="$REGION" \
  --description="AllEasystent Docker images" \
  --project="$PROJECT_ID" || echo "  (already exists)"

# ── Service Account ───────────────────────────────────────────────────────────
echo "▶ Creating service account: $SA_NAME"
gcloud iam service-accounts create "$SA_NAME" \
  --display-name="AllEasystent Service Account" \
  --project="$PROJECT_ID" || echo "  (already exists)"

SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Grant required roles
for ROLE in \
  roles/pubsub.publisher \
  roles/pubsub.subscriber \
  roles/secretmanager.secretAccessor \
  roles/aiplatform.user \
  roles/run.invoker; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SA_EMAIL" \
    --role="$ROLE" --quiet
done
echo "  Roles granted to $SA_EMAIL"

# ── Pub/Sub ───────────────────────────────────────────────────────────────────
echo "▶ Creating Pub/Sub topics and subscriptions..."
for TOPIC in incoming-messages outgoing-messages; do
  gcloud pubsub topics create "$TOPIC" --project="$PROJECT_ID" || echo "  $TOPIC already exists"
done

gcloud pubsub subscriptions create incoming-messages-sub \
  --topic=incoming-messages \
  --ack-deadline=60 \
  --message-retention-duration=1h \
  --project="$PROJECT_ID" || echo "  subscription already exists"

# ── Secret Manager ────────────────────────────────────────────────────────────
echo "▶ Creating secret placeholders in Secret Manager..."
for SECRET in \
  anthropic-api-key \
  fb-page-token \
  fb-app-secret \
  fb-verify-token \
  allegro-client-id \
  allegro-client-secret \
  infakt-api-key \
  vapid-private-key \
  vapid-public-key \
  analytics-allowed-emails; do
  gcloud secrets create "$SECRET" \
    --replication-policy=automatic \
    --project="$PROJECT_ID" 2>/dev/null || echo "  $SECRET already exists"
done

# ── Cloud Scheduler: trigger the order-monitor Cloud Run service ──────────────
# order-monitor runs as a small always-deployed Cloud Run *service* (POST
# /run), not a Cloud Run Job — Jobs cold-start a brand-new container on every
# single execution (no warm-instance reuse between runs), which at a 2-minute
# schedule made cold-start overhead the dominant cost. A service can keep a
# warm instance across back-to-back scheduler ticks like any other Cloud Run
# traffic, so the same 2-minute cadence costs a fraction as much.
#
# Run this AFTER deploy-jobs.yml has run at least once (push to main touching
# jobs/**), since gcloud run services describe needs the service to already
# exist. Auth is OIDC (--oidc-service-account-email), not OAuth — SA_EMAIL
# already holds roles/run.invoker project-wide (granted above), which covers
# invoking this service; no extra per-service IAM binding needed.
SERVICE_NAME="alleasystent-order-monitor"
SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" --project="$PROJECT_ID" --format="value(status.url)" 2>/dev/null || true)"
if [ -n "$SERVICE_URL" ]; then
  echo "▶ Creating Cloud Scheduler trigger for $SERVICE_NAME (every 2 minutes)..."
  gcloud scheduler jobs create http "${SERVICE_NAME}-trigger" \
    --location="$REGION" \
    --schedule="*/2 * * * *" \
    --uri="${SERVICE_URL}/run" \
    --http-method=POST \
    --oidc-service-account-email="$SA_EMAIL" \
    --oidc-token-audience="$SERVICE_URL" \
    --project="$PROJECT_ID" || echo "  (already exists — edit with 'gcloud scheduler jobs update http')"
else
  echo "▶ Skipping Cloud Scheduler trigger — $SERVICE_NAME doesn't exist yet."
  echo "  Push to main (or run 'gh workflow run deploy-jobs.yml') to create it, then re-run this script."
fi

echo ""
echo "✅ GCP setup complete!"
echo ""
echo "Next steps:"
echo "  1. Add secret values:  gcloud secrets versions add allegro-client-id --data-file=-  (etc.)"
echo "  2. Generate VAPID keys (Web Push was never configured — required for OS/browser"
echo "     push, not needed for the in-app Notifications panel):"
echo "       python generate_vapid_keys.py"
echo "       printf '%s' \"\$VAPID_PRIVATE_KEY_PEM\" | gcloud secrets versions add vapid-private-key --data-file=-"
echo "       printf '%s' \"\$VAPID_PUBLIC_KEY\"       | gcloud secrets versions add vapid-public-key  --data-file=-"
echo "     Then add VAPID_PRIVATE_KEY=vapid-private-key:latest,VAPID_PUBLIC_KEY=vapid-public-key:latest"
echo "     to --update-secrets, and VAPID_EMAIL=mailto:you@example.com to --update-env-vars,"
echo "     on BOTH the 'alleasystent' service (deploy-backend.yml) and the"
echo "     'alleasystent-order-monitor' service (deploy-jobs.yml)."
echo "  3. Set up the analytics dashboard (now in the alleasystent-analytics repo,"
echo "     Google Sign-In instead of a secret URL token):"
echo "       - Create an OAuth 2.0 Client ID (Web application) in GCP Console →"
echo "         APIs & Services → Credentials. Authorized JavaScript origin:"
echo "         https://mmarczyk.github.io"
echo "       - GCP secret (comma-separated allowlist of Google emails):"
echo "           printf '%s' 'you@example.com' | gcloud secrets versions add analytics-allowed-emails --data-file=-"
echo "       - GitHub repo variables on alleasystent (Settings → Secrets and variables → Actions → Variables):"
echo "           ANALYTICS_GOOGLE_CLIENT_ID = <the OAuth Client ID>"
echo "           ANALYTICS_FRONTEND_URL     = https://mmarczyk.github.io/alleasystent-analytics"
echo "       - GitHub repo variables on alleasystent-analytics:"
echo "           BACKEND_URL       = <this Cloud Run URL>"
echo "           GOOGLE_CLIENT_ID  = <the same OAuth Client ID>"
echo "  4. Build & deploy:     push to main (deploy-backend.yml / deploy-jobs.yml via GitHub Actions)"
echo "  5. Set FB webhook URL: https://YOUR_CLOUD_RUN_URL/webhook/facebook"
echo "  6. Re-run this script once the order-monitor service exists, to create its Scheduler trigger"
