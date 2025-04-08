import torch
class Config:
    # Data parameters
    max_vocab_size = 50000
    max_seq_len = 100
    
    # Model parameters
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    input_dim = 8000
    output_dim = 512
    hidden_dim = 64
    dropout = 0.3
    max_summary_length = 1384
    
    # Training parameters
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 15
    clip_grad = 5.0
    
    # RL parameters
    actor_learning_rate = 0.0001
    critic_learning_rate = 0.001
    discount_factor = 0.99
    
    # Paths
    # data_path = 'data/'
    model_path = 'models/checkpoints/'
   
    
    # Misc
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'