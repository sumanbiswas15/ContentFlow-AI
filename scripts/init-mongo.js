// MongoDB initialization script for ContentFlow AI

// Switch to the application database
db = db.getSiblingDB('contentflow_ai');

// Create application user
db.createUser({
  user: 'contentflow_user',
  pwd: 'contentflow_password',
  roles: [
    {
      role: 'readWrite',
      db: 'contentflow_ai'
    }
  ]
});

// Create collections with validation
db.createCollection('content_items', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['type', 'title', 'content', 'user_id'],
      properties: {
        type: {
          bsonType: 'string',
          enum: ['text', 'image', 'audio', 'video']
        },
        title: {
          bsonType: 'string',
          minLength: 1,
          maxLength: 200
        },
        workflow_state: {
          bsonType: 'string',
          enum: ['discover', 'create', 'transform', 'plan', 'publish', 'analyze', 'improve']
        },
        user_id: {
          bsonType: 'string'
        }
      }
    }
  }
});

db.createCollection('async_jobs', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['job_type', 'status', 'engine', 'operation', 'user_id'],
      properties: {
        job_type: {
          bsonType: 'string',
          enum: ['content_generation', 'content_transformation', 'creative_assistance', 'social_media_optimization', 'analytics_processing', 'media_generation']
        },
        status: {
          bsonType: 'string',
          enum: ['queued', 'running', 'completed', 'failed', 'cancelled']
        },
        priority: {
          bsonType: 'int',
          minimum: 1,
          maximum: 10
        }
      }
    }
  }
});

db.createCollection('users', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['email', 'username', 'hashed_password'],
      properties: {
        email: {
          bsonType: 'string',
          pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        },
        username: {
          bsonType: 'string',
          minLength: 3,
          maxLength: 50
        },
        is_active: {
          bsonType: 'bool'
        }
      }
    }
  }
});

// Create indexes for optimal performance
db.content_items.createIndex({ 'user_id': 1 });
db.content_items.createIndex({ 'type': 1 });
db.content_items.createIndex({ 'workflow_state': 1 });
db.content_items.createIndex({ 'created_at': -1 });
db.content_items.createIndex({ 'tags': 1 });
db.content_items.createIndex({ 'title': 'text', 'content': 'text' });

db.async_jobs.createIndex({ 'user_id': 1 });
db.async_jobs.createIndex({ 'status': 1 });
db.async_jobs.createIndex({ 'content_id': 1 });
db.async_jobs.createIndex({ 'created_at': -1 });
db.async_jobs.createIndex({ 'engine': 1 });
db.async_jobs.createIndex({ 'priority': 1, 'created_at': 1 });

db.users.createIndex({ 'email': 1 }, { unique: true });
db.users.createIndex({ 'username': 1 }, { unique: true });
db.users.createIndex({ 'api_keys.key_id': 1 });

print('ContentFlow AI database initialized successfully!');