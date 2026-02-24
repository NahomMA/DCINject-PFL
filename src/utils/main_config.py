CONFIG = {
    "device": "0", 
    "seed": 2024,
    
    "dataset": "cifar10",  
    
    "total_round": 300,
    "model": "resnet10",
    "model_size": 10,
    "learning_rate": 0.1,
    "client_local_step": 15,
    "client_batch": 32,
    
    "client_num": 100,
    "bad_client_num": 10,
    "select_client_num_per_round": 10,
    "client_dist": "non_iid",
    "dir_alpha": 0.5,
    "pfl": "fedbn",
    "agg_rule": "avg",
    
    "attack_type": "msba",  
    "target_label": 0,
    "poison_ratio": 0.2,
    "attack_budget": 0.03,
    "noise_pattern": "gaussian",  
}
