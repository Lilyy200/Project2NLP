import requests

# LM Studio's base URL
base_url = "http://192.168.1.68:1234"

# The model API identifier
model_name = "llama-3.2-3b-instruct"

# The endpoint for chat completions
endpoint = f"{base_url}/v1/chat/completions"

# Prepare the payload in OpenAI format
payload = {
    "model": model_name,
    "messages": [
        {"role": "user", "content": "Given the following review for an insurance company: {user_input} predict a star rating going from 1-5: "}
    ],
    # You can set additional generation parameters here:
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 1.0,
    # ...
}

# Make the POST request
response = requests.post(endpoint, json=payload)

# Parse JSON response
if response.status_code == 200:
    data = response.json()
    # data should contain something like: {"id": "...", "object": "...", "choices": [...], ...}
    # Usually the text is in data["choices"][0]["message"]["content"]
    completion_text = data["choices"][0]["message"]["content"]
    print("Model response:", completion_text)
else:
    print("Error:", response.status_code, response.text)