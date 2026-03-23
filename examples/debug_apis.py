# save as debug_apis.py in your project root
from dotenv import load_dotenv
load_dotenv()
import os, json, urllib.request

groq_key   = os.environ.get('GROQ_API_KEY')
gemini_key = os.environ.get('GEMINI_API_KEY')

print(f"Groq key loaded:   {'YES' if groq_key else 'NO'}")
print(f"Gemini key loaded: {'YES' if gemini_key else 'NO'}")

# Test Groq
if groq_key:
    payload = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10
    }).encode('utf-8')
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {groq_key}"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            result = json.loads(r.read().decode('utf-8'))
            print(f"Groq works: {result['choices'][0]['message']['content']}")
    except Exception as e:
        print(f"Groq error: {e}")

# Test Gemini
if gemini_key:
    payload = json.dumps({
        "contents": [{"parts": [{"text": "Say hello in one word."}]}]
    }).encode('utf-8')
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.0-flash:generateContent?key={gemini_key}")
    req = urllib.request.Request(url, data=payload,
          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            result = json.loads(r.read().decode('utf-8'))
            text = result['candidates'][0]['content']['parts'][0]['text']
            print(f"Gemini works: {text}")
    except Exception as e:
        print(f"Gemini error: {e}")