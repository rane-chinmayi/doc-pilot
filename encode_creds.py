import json

# Read creds.json
with open("creds.json", "r") as f:
    creds = json.load(f)

# Convert to single-line JSON string
creds_string = json.dumps(creds)

# Print the result
print(creds_string)
