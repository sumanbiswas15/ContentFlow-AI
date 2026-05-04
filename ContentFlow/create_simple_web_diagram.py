from diagrams import Diagram, Cluster
from diagrams.aws.compute import EC2, ECS
from diagrams.aws.database import RDS
from diagrams.aws.network import ELB, Route53
from diagrams.aws.storage import S3

with Diagram("Simple Web Service Architecture", show=False, filename="simple-web-service"):
    dns = Route53("DNS")
    lb = ELB("Load Balancer")
    
    with Cluster("Web Tier"):
        web_servers = [
            ECS("Web 1"),
            ECS("Web 2"),
            ECS("Web 3")
        ]
    
    with Cluster("Database"):
        db_primary = RDS("Primary DB")
        db_replica = RDS("Read Replica")
        db_primary - db_replica
    
    cache = S3("Static Assets")
    
    dns >> lb >> web_servers
    web_servers >> db_primary
    web_servers >> cache

print("Simple web service diagram generated!")
