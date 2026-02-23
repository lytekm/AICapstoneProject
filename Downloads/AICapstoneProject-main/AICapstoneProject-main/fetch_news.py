import feedparser
import urllib.request
import pandas as pd  # Import the tool for saving data
from datetime import datetime

url = "https://www.cbc.ca/cmlink/rss-business"
request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

print(f"Connecting to {url}...")

try:
    with urllib.request.urlopen(request, timeout=15) as response:
        content = response.read()
        feed = feedparser.parse(content)
        
    # Create a list to hold our data
    news_data = []

    for entry in feed.entries[:10]: # Let's grab 10 instead of 5
        news_data.append({
            "Title": entry.title,
            "Link": entry.link,
            "Date_Fetched": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # Convert the list to a DataFrame (Table)
    df = pd.DataFrame(news_data)

    # Save to a CSV file
    df.to_csv("cbc_news_data.csv", index=False)
    print("\nSuccess! Data saved to 'cbc_news_data.csv'")
    print(df.head()) # Shows the first few rows in your terminal

except Exception as e:
    print(f"Operation failed: {e}")