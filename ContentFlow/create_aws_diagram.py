from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import ECS, Lambda
from diagrams.aws.database import RDS, Dynamodb, ElastiCache
from diagrams.aws.network import APIGateway, ALB
from diagrams.aws.storage import S3
from diagrams.aws.integration import SQS, SNS, Eventbridge
from diagrams.aws.management import Cloudwatch
from diagrams.aws.ml import Bedrock, Sagemaker, Comprehend
from diagrams.aws.general import User

with Diagram("Content Intelligence Platform Architecture", show=False, direction="LR", filename="content-intelligence-platform"):
    # User/Client Layer
    user = User("Content Creator")
    
    # API Gateway
    api = APIGateway("API Gateway")
    
    # Application Layer
    with Cluster("Application Services"):
        app_lb = ALB("Application LB")
        app_servers = [
            ECS("Content Service"),
            ECS("Analytics Service"),
            ECS("AI Service")
        ]
    
    # AI/ML Layer
    with Cluster("AI/ML Processing"):
        bedrock = Bedrock("Bedrock AI")
        sagemaker = Sagemaker("SageMaker")
        comprehend = Comprehend("Comprehend")
    
    # Data Processing
    with Cluster("Data Processing"):
        lambda_proc = Lambda("Content Processor")
        lambda_analytics = Lambda("Analytics Processor")
        eventbridge = Eventbridge("Event Bus")
    
    # Storage Layer
    with Cluster("Storage"):
        s3_content = S3("Content Storage")
        s3_analytics = S3("Analytics Data")
    
    # Database Layer
    with Cluster("Databases"):
        rds = RDS("Content DB")
        dynamodb = Dynamodb("Metadata Store")
        elasticache = ElastiCache("Cache Layer")
    
    # Messaging & Queue
    with Cluster("Messaging"):
        sqs = SQS("Processing Queue")
        sns = SNS("Notifications")
    
    # Monitoring
    cloudwatch = Cloudwatch("CloudWatch")
    
    # Flow
    user >> api >> app_lb >> app_servers
    
    app_servers >> elasticache
    app_servers >> rds
    app_servers >> dynamodb
    app_servers >> s3_content
    
    app_servers >> sqs >> lambda_proc
    lambda_proc >> bedrock
    lambda_proc >> sagemaker
    lambda_proc >> comprehend
    
    lambda_proc >> eventbridge >> lambda_analytics
    lambda_analytics >> s3_analytics
    
    app_servers >> sns
    
    [app_servers[0], lambda_proc, lambda_analytics] >> cloudwatch

print("Diagram generated successfully!")
