import feedparser
import urllib.request

url = "https://www.cbc.ca/cmlink/rss-business"

# 1. We tell the website we are a 'Browser' not a 'Bot'
request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

print(f"Attempting to connect to {url}...")

try:
    # 2. Fetch the data manually first to bypass basic blocks
    with urllib.request.urlopen(request, timeout=15) as response:
        content = response.read()
        feed = feedparser.parse(content)
        
    # 3. Print the results
    if not feed.entries:
        print("Connected, but no news found.")
    else:
        for entry in feed.entries[:5]:
            print(f"\nHeadline: {entry.title}")
            print(f"Link: {entry.link}")

except Exception as e:
    print(f"Connection failed: {e}")