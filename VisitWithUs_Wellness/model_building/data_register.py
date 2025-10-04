from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Define your dataset repo
repo_id = "sureshsharma4747/Customer-Purchase-Prediction"
repo_type="dataset"

# 1. Create dataset repo if not exists
try:
    api.repo_info(repo_id=repo_id, repo_type="dataset")
    print(f"✅ Dataset repo already exists: {repo_id}")
except:
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
    print(f"✅ Created new dataset repo: {repo_id}")

# 2. Upload local folder
api.upload_folder(
    folder_path="VisitWithUs_Wellness/data",
    repo_id=repo_id,
    repo_type="dataset"
)

# 3. Print Hugging Face dataset URL
print("🚀 Upload finished! Check your dataset here:")
print(f"https://huggingface.co/datasets/{repo_id}")
