import requests

api_key = "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"
headers = {"Authorization": f"Bearer {api_key}"}
url = "https://integrate.api.nvidia.com/v1/status"  # Replace with a valid endpoint

response = requests.get(url, headers=headers)

# Check the response and content
if response.status_code == 200:
    print("API key is valid. Response:", response.json())
else:
    print(f"Failed with status code {response.status_code}. Response text: {response.text}")
