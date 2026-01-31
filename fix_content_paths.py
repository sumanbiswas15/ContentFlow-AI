"""
Script to fix content paths in database to use actual files from storage folder.
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

# Actual files that exist in storage
ACTUAL_FILES = {
    'image': [
        '/storage/images/thumbnails/442216019eb0a351.png',
        '/storage/images/thumbnails/4f6c2bfc316d91e2.png',
        '/storage/images/thumbnails/704499927c7ebf18.png',
        '/storage/images/thumbnails/748aaf8924c05987.png',
        '/storage/images/thumbnails/cdfb3df31f859773.png',
        '/storage/images/thumbnails/dda7fd3172c23612.png',
    ],
    'video': [
        '/storage/videos/short_form/6febfadc31b88956.mp4',
    ],
    'audio': [
        '/storage/audio/music/a449bf2762b5317b.mp3',
    ]
}

async def fix_content_paths():
    """Fix content paths to use actual files."""
    # Connect to MongoDB
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client.contentflow_ai
    
    # Get all content items
    cursor = db.content_items.find({})
    content_items = await cursor.to_list(length=None)
    
    print(f"Found {len(content_items)} content items")
    
    updated_count = 0
    for item in content_items:
        content_type = item.get('type')
        current_content = item.get('content', '')
        
        # Check if content references a non-existent file
        if isinstance(current_content, str) and ('T_logo' in current_content or 'images/' in current_content or 'videos/' in current_content or 'audio/' in current_content):
            # Get an actual file for this type
            if content_type in ACTUAL_FILES and ACTUAL_FILES[content_type]:
                new_content = ACTUAL_FILES[content_type][updated_count % len(ACTUAL_FILES[content_type])]
                
                # Update the content
                await db.content_items.update_one(
                    {'_id': item['_id']},
                    {'$set': {'content': new_content}}
                )
                
                print(f"Updated {item['_id']}: {current_content} -> {new_content}")
                updated_count += 1
    
    print(f"\nUpdated {updated_count} content items")
    client.close()

if __name__ == "__main__":
    asyncio.run(fix_content_paths())
