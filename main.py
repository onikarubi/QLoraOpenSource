from train import ModelTrainer
from dataset_manager import DatasetManager

if __name__ == '__main__':
    data_path = './sample.jsonl'
    manager = DatasetManager(data_path, format='chat_openai')
