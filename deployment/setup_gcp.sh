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
  roles/aiplatform.user; do
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
  infakt-api-key; do
  gcloud secrets create "$SECRET" \
    --replication-policy=automatic \
    --project="$PROJECT_ID" 2>/dev/null || echo "  $SECRET already exists"
done

echo ""
echo "✅ GCP setup complete!"
echo ""
echo "Next steps:"
echo "  1. Add secret values:  gcloud secrets versions add anthropic-api-key --data-file=-"
echo "  2. Build & deploy:     gcloud builds submit --config=cloudbuild.yaml"
echo "  3. Set FB webhook URL: https://YOUR_CLOUD_RUN_URL/webhook/facebook"
