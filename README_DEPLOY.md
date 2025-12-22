1) Build the Docker image locally:
   docker build -t disaster-tweets-api:1.0 .

2) Run locally (replace HF_TOKEN):
   docker run --rm -e HF_TOKEN="hf_xxx" -e HF_REPO_ID="sakibalfahim/disaster-tweets-bert" -p 8080:8080 disaster-tweets-api:1.0

3) Example curl:
   curl -X POST "http://localhost:8080/predict" -H "Content-Type: application/json" -d '{"text": "Huge fire near the forest"}'

4) To deploy to Google Cloud Run:
   - gcloud builds submit --tag gcr.io/<PROJECT-ID>/disaster-tweets-api
   - gcloud run deploy disaster-tweets-api --image gcr.io/<PROJECT-ID>/disaster-tweets-api --platform managed --region <REGION> --allow-unauthenticated --set-env-vars HF_TOKEN=hf_xxx,HF_REPO_ID=sakibalfahim/disaster-tweets-bert
