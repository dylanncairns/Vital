# Runbook

## Start API
```bash
uvicorn api.main:app --reload
```

## Run Worker (Continuous)
```bash
python3 -m api.job_worker --limit 100 --max-papers-per-query 5
```

## Run Worker (One Batch)
```bash
python3 -m api.job_worker --once --limit 100 --max-papers-per-query 5
```

## Recompute Insights (One User - Replace user_id with user's id)
```bash
curl -X POST "http://127.0.0.1:8000/insights/recompute" \
  -H "Content-Type: application/json" \
  -d '{"user_id":1,"online_enabled":false,"max_papers_per_query":5}'
```

## Recompute Insights (All Users)
```bash
for uid in $(psql "$DATABASE_URL" -Atc "SELECT id FROM users;"); do
  curl -s -X POST "http://127.0.0.1:8000/insights/recompute" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":$uid,\"online_enabled\":false,\"max_papers_per_query\":5}" >/dev/null
  echo "recomputed user $uid"
done
```

## RAG Sync (Manual Evidence Acquisition)
```bash
curl -X POST "http://127.0.0.1:8000/rag/sync" \
  -H "Content-Type: application/json" \
  -d '{"user_id":1,"online_enabled":true,"max_papers_per_query":8}'
```

## Process Background Jobs via API
```bash
curl -X POST "http://127.0.0.1:8000/jobs/process" \
  -H "Content-Type: application/json" \
  -d '{"limit":100,"max_papers_per_query":5}'
```

## Citation Audit (Run Immediately)
```bash
curl -X POST "http://127.0.0.1:8000/citations/audit" \
  -H "Content-Type: application/json" \
  -d '{"limit":300,"delete_missing":true}'
```

## Citation Audit (Enqueue for Worker)
```bash
curl -X POST "http://127.0.0.1:8000/citations/audit/enqueue" \
  -H "Content-Type: application/json" \
  -d '{"user_id":1,"limit":300,"delete_missing":true}'
```

## Background Jobs Status
```bash
psql "$DATABASE_URL" -c "SELECT status, COUNT(*) FROM background_jobs GROUP BY status;"
```

## Supported Insights Check
```bash
psql "$DATABASE_URL" -c "SELECT id, item_id, symptom_id, final_score, display_decision_reason FROM insights WHERE display_decision_reason='supported' ORDER BY id DESC LIMIT 20;"
```
