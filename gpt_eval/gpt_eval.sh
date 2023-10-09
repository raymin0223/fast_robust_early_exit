python api_request_parallel_processor.py \
  --requests_filepath ./inputs/likert_free.jsonl \
  --save_filepath ./outputs/likert_free.jsonl \
  --request_url https://api.openai.com/v1/chat/completions \
  --api_key API_KEY \
  --max_requests_per_minute 2500 \
  --max_tokens_per_minute 200000 \
  --max_attempts 20 \
  --logging_level 20