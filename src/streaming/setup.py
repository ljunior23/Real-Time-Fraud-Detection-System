
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
import json
import pandas as pd
import time
from datetime import datetime

class FraudStreamSetup:
    def __init__(self, bootstrap_servers='localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.topic_name = 'fraud-transactions'
    
    def create_topic(self):
        """Create Kafka topic for transactions"""
        admin_client = KafkaAdminClient(
            bootstrap_servers=self.bootstrap_servers
        )
        
        topic = NewTopic(
            name=self.topic_name,
            num_partitions=3,
            replication_factor=1
        )
        
        try:
            admin_client.create_topics([topic])
            print(f"Topic '{self.topic_name}' created successfully")
        except Exception as e:
            print(f"Topic already exists or error: {e}")
    
    def simulate_stream(self, data_path, delay=0.1):
        """Simulate streaming transactions from CSV"""
        
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Load data
        df = pd.read_csv(data_path)
        
        print(f"Starting to stream {len(df)} transactions...")
        print(f"Delay between transactions: {delay}s")
        
        for idx, row in df.iterrows():
            # Create transaction message
            transaction = {
                'transaction_id': idx,
                'timestamp': datetime.now().isoformat(),
                'features': row.drop('Class').to_dict(),
                'actual_label': int(row['Class'])  # For testing only
            }
            
            # Send to Kafka
            producer.send(self.topic_name, transaction)
            
            if (idx + 1) % 1000 == 0:
                print(f"Sent {idx + 1} transactions...")
            
            time.sleep(delay)
        
        producer.flush()
        print(f"Finished streaming {len(df)} transactions")
    
    def test_consumer(self, max_messages=10):
        """Test consuming messages from Kafka"""
        
        consumer = KafkaConsumer(
            self.topic_name,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            group_id='test-consumer'
        )
        
        print(f"Consuming up to {max_messages} messages...")
        
        count = 0
        for message in consumer:
            print(f"\nMessage {count + 1}:")
            print(f"  Partition: {message.partition}")
            print(f"  Offset: {message.offset}")
            print(f"  Transaction ID: {message.value['transaction_id']}")
            print(f"  Timestamp: {message.value['timestamp']}")
            
            count += 1
            if count >= max_messages:
                break
        
        consumer.close()

# Usage example (will be used in Week 2)
if __name__ == "__main__":
    setup = FraudStreamSetup()
    
    # Create topic
    setup.create_topic()
    
    # Test with small sample
    print("\nTesting with 100 transactions...")
    # setup.simulate_stream('../data/creditcard.csv', delay=0.01)
    
    # Test consumer
    # setup.test_consumer(max_messages=5)