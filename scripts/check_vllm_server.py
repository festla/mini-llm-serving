import requests


url = "http://127.0.0.1:8000/health"

r = requests.get(
    url,
    timeout=5
)

print(r.status_code)
print(r.text)
