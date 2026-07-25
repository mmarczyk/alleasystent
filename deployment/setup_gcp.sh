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
  firestore.googleapis.com \
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
  roles/datastore.user \
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

# ── Firestore ─────────────────────────────────────────────────────────────────
echo "▶ Initializing Firestore (native mode)..."
gcloud firestore databases create \
  --location="$REGION" \
  --project="$PROJECT_ID" || echo "  (already exists)"

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
  vapid-public-key; do
  gcloud secrets create "$SECRET" \
    --replication-policy=automatic \
    --project="$PROJECT_ID" 2>/dev/null || echo "  $SECRET already exists"
done

# ── Cloud Scheduler: trigger the order-monitor Cloud Run Job ──────────────────
# Run this AFTER deploy-jobs.yml has run at least once (push to main touching
# jobs/**), since gcloud run jobs describe needs the job to already exist.
JOB_NAME="alleasystent-order-monitor"
if gcloud run jobs describe "$JOB_NAME" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  echo "▶ Creating Cloud Scheduler trigger for $JOB_NAME (every 2 minutes)..."
  JOB_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run"
  gcloud scheduler jobs create http "${JOB_NAME}-trigger" \
    --location="$REGION" \
    --schedule="*/2 * * * *" \
    --uri="$JOB_URI" \
    --http-method=POST \
    --oauth-service-account-email="$SA_EMAIL" \
    --project="$PROJECT_ID" || echo "  (already exists — edit schedule with 'gcloud scheduler jobs update http')"
else
  echo "▶ Skipping Cloud Scheduler trigger — $JOB_NAME doesn't exist yet."
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
echo "     'alleasystent-order-monitor' job (deploy-jobs.yml)."
echo "  3. Build & deploy:     push to main (deploy-backend.yml / deploy-jobs.yml via GitHub Actions)"
echo "  4. Set FB webhook URL: https://YOUR_CLOUD_RUN_URL/webhook/facebook"
echo "  5. Re-run this script once the order-monitor job exists, to create its Scheduler trigger"
