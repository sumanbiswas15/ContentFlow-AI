from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import ECS, Lambda, Fargate
from diagrams.aws.network import ALB, CloudFront
from diagrams.aws.database import DocumentDB
from diagrams.aws.storage import S3
from diagrams.aws.integration import SQS, SNS
from diagrams.aws.ml import Bedrock
from diagrams.aws.management import Cloudwatch
from diagrams.onprem.client import Users

with Diagram("ContentFlow AI Architecture", show=False, direction="LR"):
    users = Users("Content Creators")
    
    with Cluster("Frontend"):
        cf = CloudFront("CloudFront CDN")
        s3_frontend = S3("React App\n(Static Hosting)")
    
    alb = ALB("Application\nLoad Balancer")
    
    with Cluster("API Layer"):
        api = ECS("FastAPI\nBackend")
    
    with Cluster("AI Engines"):
        orchestrator = Lambda("Orchestrator")
        
        with Cluster("Content Generation"):
            text_engine = Lambda("Text\nIntelligence")
            image_engine = Lambda("Image\nGeneration")
            video_engine = Lambda("Video\nPipeline")
            audio_engine = Lambda("Audio\nGeneration")
        
        with Cluster("Planning & Analytics"):
            creative = Lambda("Creative\nAssistant")
            social = Lambda("Social Media\nPlanner")
            analytics = Lambda("Discovery\nAnalytics")
    
    with Cluster("Job Processing"):
        queue = SQS("Job Queue")
        processor = Lambda("Job\nProcessor")
    
    mongodb = DocumentDB("MongoDB\n(Users, Content, Jobs)")
    
    with Cluster("Storage"):
        s3_images = S3("Images")
        s3_videos = S3("Videos")
        s3_audio = S3("Audio")
    
    bedrock = Bedrock("Amazon Bedrock\n(Gemini Models)")
    cloudwatch = Cloudwatch("CloudWatch\nMonitoring")
    
    # User flow
    users >> cf >> s3_frontend >> alb >> api
    
    # Orchestration
    api >> orchestrator
    orchestrator >> [text_engine, image_engine, video_engine, audio_engine]
    orchestrator >> [creative, social, analytics]
    
    # Job processing
    api >> queue >> processor
    processor >> [text_engine, image_engine, video_engine, audio_engine]
    
    # AI integration
    [text_engine, image_engine, video_engine, audio_engine, creative, social, analytics] >> bedrock
    
    # Database
    [api, processor, orchestrator] >> mongodb
    
    # Storage
    image_engine >> s3_images
    video_engine >> s3_videos
    audio_engine >> s3_audio
    api >> [s3_images, s3_videos, s3_audio]
    
    # Monitoring
    [api, processor, orchestrator] >> cloudwatch

print("ContentFlow AI diagram generated successfully!")
