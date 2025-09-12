"""
Quick test for BigQuery AI Resume Matcher setup
"""

import os
import sys
from pathlib import Path

# Set environment variables first
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\chand computer\Desktop\My WorkSpace4\Kaggel\divine-catalyst-459423-j5-6b5f13aeff7c.json'
os.environ['GOOGLE_CLOUD_PROJECT'] = 'divine-catalyst-459423-j5'

print("🔧 Quick BigQuery Test")
print("=" * 30)

try:
    from google.cloud import bigquery
    print("✅ BigQuery library imported")
    
    # Test client creation
    client = bigquery.Client()
    print(f"✅ Client created for project: {client.project}")
    
    # Test simple query
    query = "SELECT 'Hello BigQuery!' as message, CURRENT_TIMESTAMP() as timestamp"
    result = client.query(query).result()
    
    for row in result:
        print(f"✅ Query successful: {row.message}")
        print(f"   Timestamp: {row.timestamp}")
    
    print("\n🎉 BigQuery connection working!")
    print("🚀 Ready to run the full pipeline!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Possible solutions:")
    print("   • Check internet connection")
    print("   • Verify BigQuery API is enabled")
    print("   • Ensure service account has proper permissions")
