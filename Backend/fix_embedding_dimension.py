"""
Fix ChromaDB embedding dimension mismatch
Run this script to reset the database when switching embedding models
"""
import os
import shutil
from datetime import datetime

print("="*60)
print("ChromaDB Dimension Fix Script")
print("="*60)

# Check if chroma_db exists
if os.path.exists("chroma_db"):
    print("\n⚠️  WARNING: This will delete your existing ChromaDB database!")
    print("   All uploaded documents will need to be re-uploaded.")
    
    response = input("\nContinue? (yes/no): ")
    
    if response.lower() == "yes":
        # Create backup name with timestamp
        backup_name = f"chroma_db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Rename old database
            shutil.move("chroma_db", backup_name)
            print(f"\n✓ Old database backed up to: {backup_name}")
            
            # Create new empty directory
            os.makedirs("chroma_db", exist_ok=True)
            print("✓ New database directory created")
            
            print("\n" + "="*60)
            print("SUCCESS!")
            print("="*60)
            print("\nNext steps:")
            print("1. Restart your backend server")
            print("2. Re-upload your PDF documents")
            print("3. The new embeddings will use 3072 dimensions")
            print("\nOld database backup location:")
            print(f"   {os.path.abspath(backup_name)}")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            print("\nTroubleshooting:")
            print("1. Stop the backend server (Ctrl+C)")
            print("2. Run this script again")
            print("3. Or manually delete the chroma_db folder")
    else:
        print("\nOperation cancelled.")
else:
    print("\n✓ No existing database found. Creating new one...")
    os.makedirs("chroma_db", exist_ok=True)
    print("✓ Database directory created")
    print("\nYou can now start the server and upload documents.")

print("\n" + "="*60)
