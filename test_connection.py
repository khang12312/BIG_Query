"""
Test BigQuery connection and setup for AI-Powered Resume Matcher
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.bigquery_client import BigQueryAIClient
from src.config import get_config

def test_bigquery_connection():
    """Test BigQuery connection and basic operations"""
    print("🔧 Testing BigQuery AI Connection...")
    print("=" * 50)
    
    try:
        # Set environment variables
        config = get_config('bigquery')
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config['credentials_path']
        os.environ['GOOGLE_CLOUD_PROJECT'] = config['project_id']
        
        print(f"📋 Configuration:")
        print(f"   • Project ID: {config['project_id']}")
        print(f"   • Dataset: {config['dataset_id']}")
        print(f"   • Location: {config['location']}")
        print(f"   • Credentials: {config['credentials_path']}")
        print()
        
        # Initialize client
        print("🚀 Initializing BigQuery AI client...")
        client = BigQueryAIClient()
        
        # Test basic connection
        print("🔍 Testing connection...")
        datasets = list(client.client.list_datasets())
        print(f"✅ Connection successful! Found {len(datasets)} datasets")
        
        # Create dataset if needed
        print("📊 Creating dataset and tables...")
        client.create_dataset_if_not_exists()
        client.create_tables()
        print("✅ Dataset and tables ready!")
        
        # Test query capability
        print("🔍 Testing query capability...")
        test_query = f"""
        SELECT 
            '{config['project_id']}' as project_id,
            '{config['dataset_id']}' as dataset_id,
            CURRENT_TIMESTAMP() as test_time
        """
        
        result = client.client.query(test_query).to_dataframe()
        print("✅ Query test successful!")
        print(f"   • Test time: {result.iloc[0]['test_time']}")
        
        print("\n🎉 BigQuery AI setup complete and ready!")
        print("🚀 You can now run the main pipeline: python main.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing connection: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   • Ensure your service account has BigQuery Admin permissions")
        print("   • Check that BigQuery API is enabled in your project")
        print("   • Verify the credentials file path is correct")
        return False

if __name__ == "__main__":
    test_bigquery_connection()
