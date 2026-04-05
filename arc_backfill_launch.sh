#!/bin/bash
# arc_backfill_launch.sh — Launch a vast.ai backfill worker
#
# Usage:
#   ./arc_backfill_launch.sh <offer_id> <start_idx> <end_idx>
#   ./arc_backfill_launch.sh 32781617 0 500
#   ./arc_backfill_launch.sh auto 0 500      # auto-pick cheapest offer

set -e
source /home/jeff/arc/.env

OFFER_ID=${1:?Usage: $0 <offer_id|auto> [start] [end]}
START_IDX=${2:-}
END_IDX=${3:-}

# Auto-pick cheapest offer
if [ "$OFFER_ID" = "auto" ]; then
    OFFER_ID=$(~/.local/bin/vastai search offers \
        "gpu_ram>=8 inet_up>200 disk_space>30 cuda_vers>=12.0 reliability>0.95" \
        --order dph_total --limit 1 --raw 2>/dev/null \
        | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['id'])")
    echo "Auto-selected offer: $OFFER_ID"
fi

# Build onstart script that bakes in credentials
ONSTART_CMD="$(cat <<ONSTART_EOF
#!/bin/bash
set -e

# ── Baked credentials ──
export R2_ENDPOINT='${R2_ENDPOINT}'
export R2_ACCESS_KEY='${R2_ACCESS_KEY}'
export R2_SECRET_KEY='${R2_SECRET_KEY}'
export R2_BUCKET='${R2_BUCKET}'
export START_IDX='${START_IDX}'
export END_IDX='${END_IDX}'

# Save for worker process
cat > /root/.worker_env <<'ENVEOF'
export R2_ENDPOINT='${R2_ENDPOINT}'
export R2_ACCESS_KEY='${R2_ACCESS_KEY}'
export R2_SECRET_KEY='${R2_SECRET_KEY}'
export R2_BUCKET='${R2_BUCKET}'
export START_IDX='${START_IDX}'
export END_IDX='${END_IDX}'
ENVEOF
chmod 600 /root/.worker_env

# ── Download and run bootstrap ──
pip install -q boto3
python3 -c "
import boto3, os
s3 = boto3.client('s3', endpoint_url=os.environ['R2_ENDPOINT'],
    aws_access_key_id=os.environ['R2_ACCESS_KEY'],
    aws_secret_access_key=os.environ['R2_SECRET_KEY'])
s3.download_file(os.environ.get('R2_BUCKET','arc-cloud'),
    'backfill/scripts/arc_backfill_bootstrap.sh',
    '/workspace/bootstrap.sh')
print('[onstart] Downloaded bootstrap script')
"
bash /workspace/bootstrap.sh
ONSTART_EOF
)"

echo "Launching vast.ai instance..."
echo "  Offer:  $OFFER_ID"
echo "  Range:  [$START_IDX, $END_IDX)"

# Write onstart to temp file (vast.ai reads it)
ONSTART_FILE=$(mktemp /tmp/backfill_onstart_XXXXX.sh)
echo "$ONSTART_CMD" > "$ONSTART_FILE"

RESULT=$(~/.local/bin/vastai create instance "$OFFER_ID" \
    --image pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime \
    --disk 25 --ssh --direct \
    --label "backfill-${START_IDX}-${END_IDX}" \
    --onstart "$ONSTART_FILE" \
    --raw 2>&1)

rm -f "$ONSTART_FILE"

echo "$RESULT"
INSTANCE_ID=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('new_contract','FAILED'))" 2>/dev/null || echo "FAILED")
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Monitor:     vastai logs $INSTANCE_ID"
echo "SSH:         vastai ssh $INSTANCE_ID"
echo "Destroy:     vastai destroy instance $INSTANCE_ID"
