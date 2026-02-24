import torch, torchvision, argparse
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from functools import partial, wraps
from random import shuffle
import datetime

from src.federated.server import BasicServer
from src.federated.client import BasicClient, PoisonClient
from src.federated.fl_process import basic_fl_process
from src.federated.event_emitter import *
from src.federated.pfl import use_fedbn

from src.models.resnet import get_resnet
from src.attacks.dcinject_trigger import DCINJECTTrigger
from src.utils.utils import random_select, evaluate_accuracy, client_inner_dirichlet_partition, set_random_seed


# from src.defenses.ibau.ibau_defense import run_ibau_defense


def time_experiment(func): 
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        result = func(*args, **kwargs)
        time_taken = (datetime.datetime.now() - start).total_seconds() / 60
        return result + (time_taken,)
    return wrapper

def setup_dcinject_attack(clients, server, target_label, poison_ratio,attack_budget,noise_pattern, trigger_type="frequency"):   
    trigger = DCINJECTTrigger(server.global_model.device, attack_budget=attack_budget, noise_pattern=noise_pattern)

    def dcinject_poison_func(data, label, target_label=target_label, poison_ratio=poison_ratio):
        if trigger_type == "frequency":
            return trigger.apply_trigger_batch_frequency(data, label, target_label, poison_ratio)
        elif trigger_type == "adaptive_frequency":
            return trigger.apply_trigger_batch_adaptive_frequency(data, label, target_label, poison_ratio)
        else:
            return trigger.apply_trigger_batch(data, label, target_label, poison_ratio)

    for client in clients:
        if "Poison" in type(client).__name__:
            client.poison_func = partial(dcinject_poison_func, 
                                       target_label=target_label, 
                                       poison_ratio=poison_ratio)
            eval_func = partial(dcinject_poison_func, 
                              target_label=target_label, 
                              poison_ratio=1.0)
    
    return eval_func

def load_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--total_round", type=int, default=400)
    parser.add_argument("--client_num", type=int, default=100)
    parser.add_argument("--bad_client_num", type=int, default=10 )
    parser.add_argument("--select_client_num_per_round", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--dir_alpha", type=float, default=0.5)
    parser.add_argument("--client_local_step", type=int, default=15)
    parser.add_argument("--client_batch", type=int, default=32)
    parser.add_argument("--pfl", type=str, default="fedbn")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--poison_ratio", type=float, default=0.2)
    parser.add_argument("--attack_budget", type=float, default=0.03)
    parser.add_argument("--noise_pattern", type=str, default="gaussian")
    parser.add_argument("--agg_rule", type=str, default="avg")
    parser.add_argument("--defense", action="store_true", help="Enable defense")
    parser.add_argument("--no_defense", action="store_true", help="Disable defense (overrides --defense)")
    parser.add_argument("--defense_rounds", type=int, default=3, help="Number of defense rounds")
    parser.add_argument("--defense_lr", type=float, default=0.001, help="Defense learning rate")
    parser.add_argument("--attack_type", type=str, default=None, help="Specify the attack to run")
    parser.add_argument("--trigger_type", type=str, default=None, help="Specify trigger type for  the attack")
    return parser.parse_args()

def load_dataset(dataset_name):
    if dataset_name.lower() == "gtsrb":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
                                                   torchvision.transforms.Resize((32, 32))])
    else:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    dataset_configs = {
        "cifar10": (torchvision.datasets.CIFAR10, 10, 'targets'),
        "cifar100": (torchvision.datasets.CIFAR100, 100, 'targets'),
        "svhn": (torchvision.datasets.SVHN, 10, 'labels'),
        "gtsrb": (torchvision.datasets.GTSRB, 43, 'targets')
    }
    
    dataset_class, num_classes, label_attr = dataset_configs[dataset_name.lower()]
    
    if dataset_name.lower() == "svhn":
        train_dataset = dataset_class('../data', split='train', download=True, transform=transform)
        test_dataset = dataset_class('../data', split='test', download=True, transform=transform)
    elif dataset_name.lower() == "gtsrb":
        train_dataset = dataset_class('../data', split='train', download=True, transform=transform)
        test_dataset = dataset_class('../data', split='test', download=True, transform=transform)
    else:
        train_dataset = dataset_class('../data', train=True, download=True, transform=transform)
        test_dataset = dataset_class('../data', train=False, download=True, transform=transform)
        
    if dataset_name.lower() == "gtsrb":
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]
    else:
        train_labels = getattr(train_dataset, label_attr)
        test_labels = getattr(test_dataset, label_attr)
    
    
    return train_dataset, test_dataset, train_labels, test_labels, num_classes

def setup_clients(args, train_dataset, test_dataset, train_labels, test_labels, num_classes, device):
    
    client_optimizer = partial(torch.optim.SGD, lr=args.learning_rate)
    
    # Data partitioning
    client_sample_nums = [len(train_dataset) // args.client_num] * args.client_num
    class_priors = np.random.dirichlet([args.dir_alpha] * num_classes, args.client_num)
    
    client_train_indices = client_inner_dirichlet_partition(
        train_labels, args.client_num, num_classes, args.dir_alpha, client_sample_nums, class_priors)
    client_test_indices = client_inner_dirichlet_partition(
        test_labels, args.client_num, num_classes, args.dir_alpha, 
        [len(test_dataset) // args.client_num] * args.client_num, class_priors)
    
    # Create dataloaders
    train_loaders = [torch.utils.data.DataLoader(train_dataset, batch_size=args.client_batch,
                     sampler=SubsetRandomSampler(client_train_indices[i]), drop_last=True) 
                     for i in range(args.client_num)]
    test_loaders = [torch.utils.data.DataLoader(test_dataset, batch_size=args.client_batch,
                    sampler=SubsetRandomSampler(client_test_indices[i]), drop_last=True) 
                    for i in range(args.client_num)]
    
    # Create clients
    clients = [BasicClient(get_resnet(size=10, num_classes=num_classes).to(device),
                          train_loaders[i], test_loaders[i], torch.nn.CrossEntropyLoss(), client_optimizer)
               for i in range(args.client_num - args.bad_client_num)]
    
    clients.extend([PoisonClient(get_resnet(size=10, num_classes=num_classes).to(device),
                                train_loaders[i], test_loaders[i], torch.nn.CrossEntropyLoss(), 
                                client_optimizer, poison_func=None)
                   for i in range(args.client_num - args.bad_client_num, args.client_num)])
    
    shuffle(clients)
    for idx, client in enumerate(clients):
        client.local_model.device = device
        client.cid = idx
        if not hasattr(client, 'client_info'): 
            client.client_info = {}
        client.client_info['base_idx'] = idx  
        client.client_info['ibau_train_indices'] = client_train_indices[idx]
        client.client_info['ibau_test_indices'] = client_test_indices[idx]        
        
        # 50% test, 50% unl        
        total_samples = len(client_test_indices[idx])
        test_split = int(0.5 * total_samples)        
        
        client.client_info['test_indices'] = client_test_indices[idx][:test_split]                    
        client.client_info['att_val_indices'] = client_test_indices[idx][test_split:]   
        client.client_info['unl_indices']  =   client_test_indices[idx][test_split:]                 
        
        client.client_info['ibau_datasets'] = test_dataset            
        
    
    return clients

@time_experiment
def run_experiment(args, attack_type, clients, num_classes, device, trigger_type=None):
    
    server = BasicServer(get_resnet(size=10, num_classes=num_classes).to(device))
    server.global_model.device = device
    server.agg_rule = args.agg_rule
    
    if args.pfl == "fedbn":
        use_fedbn(server)
    
    # Setup attack    
    poison_func = setup_dcinject_attack(clients, server, args.target_label, args.poison_ratio, 
                                       args.attack_budget, args.noise_pattern, trigger_type)
    
    # Run federated learning
    basic_fl_process(server, clients, local_steps=args.client_local_step, 
                    training_rounds=args.total_round,
                    select_rule=partial(random_select, nums=args.select_client_num_per_round))
    
    # # Save models and images for DCInject attacks
    # if attack_type == "dcinject":
    #     save_models_and_images(clients, poison_func, trigger_type or "spatial")
    
    # # Print header for results table
    # print(f"\nDetailed results for {attack_type} attack:")
    # print(f"{'Client ID':<10} {'Type':<12} {'Acc(Before)':<12} {'ASR(Before)':<12} {'Acc(After)':<12} {'ASR(After)':<12}")
    # print("-" * 80)
    
    accuracies = []
    asrs = []
    defended_accs = []
    defended_asrs = []
    
    for c in clients:
        # Before defense
        acc = evaluate_accuracy(c.local_model, c.test_dataloader)
        asr = evaluate_accuracy(c.local_model, c.test_dataloader, poison_func)
        
       
        # if args.defense:
        #     _, _, _, def_acc, def_asr = run_ibau_defense(
        #         c, poison_func, device, n_rounds=args.defense_rounds, 
        #         lr=args.defense_lr, num_classes=num_classes)
        # else:
        def_acc, def_asr = acc, asr  
        
        client_type = "Poison" if "Poison" in type(c).__name__ else "Normal"
        print(f"{c.cid:<10} {client_type:<12} {acc:<12.2f} {asr:<12.2f} {def_acc:<12.2f} {def_asr:<12.2f}")
        
        accuracies.append(acc)
        asrs.append(asr)
        defended_accs.append(def_acc)
        defended_asrs.append(def_asr)
    
    def_acc_tensor, def_asr_tensor = torch.tensor(defended_accs), torch.tensor(defended_asrs)
    acc_tensor, asr_tensor = torch.tensor(accuracies), torch.tensor(asrs)
    return (acc_tensor.mean().item(), acc_tensor.std().item(), 
            asr_tensor.mean().item(), asr_tensor.std().item(),
            def_acc_tensor.mean().item(), def_acc_tensor.std().item(), 
            def_asr_tensor.mean().item(), def_asr_tensor.std().item())

def save_models_and_images(clients, poison_func, trigger_type):
    import os
    run_dir = f"./malicious_models/{trigger_type}"
    img_dir = f"./malicious_images/{trigger_type}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    for client in clients:
        if "Poison" in type(client).__name__:
            # Save model
            model_path = os.path.join(run_dir, f"client_{client.cid}_final.pt")
            torch.save(client.local_model.state_dict(), model_path)
            
            # Save sample images
            try:
                data, labels = next(iter(client.test_dataloader))
                poisoned_data, poisoned_labels = poison_func(data[:3], labels[:3])
                for i in range(3):
                    torchvision.utils.save_image(data[i], 
                        os.path.join(img_dir, f"client_{client.cid}_orig_{i}.png"), normalize=True)
                    torchvision.utils.save_image(poisoned_data[i],
                        os.path.join(img_dir, f"client_{client.cid}_poison_{i}.png"), normalize=True)
            except: pass

# def evaluate_with_defense(clients, poison_func, device,num_classes):
#     defended_accs, defended_asrs = [], []
#     for client in clients:
#         _, _, _, def_acc, def_asr = run_ibau_defense(
#             client, poison_func, device, n_rounds=300, lr=0.001,num_classes=num_classes)
#         defended_accs.append(def_acc)
#         defended_asrs.append(def_asr)
    
#     def_acc_tensor, def_asr_tensor = torch.tensor(defended_accs), torch.tensor(defended_asrs)
#     return def_acc_tensor.mean().item(), def_acc_tensor.std().item(), def_asr_tensor.mean().item(), def_asr_tensor.std().item()


def print_results(results, defense_enabled=False):
    print("\nFinal Summary:")
    print("=" * 100)
    if defense_enabled:
        print(f"{'Dataset':<12} {'Attack':<16} {'Accuracy':<15} {'ASR':<15} {'Def_Acc':<15} {'Def_ASR':<15} {'Time':<12}")
        print("-" * 100)
        for dataset, attacks in results.items():
            for attack, metrics in attacks.items():
                acc, acc_std = metrics["accuracy"]
                asr, asr_std = metrics["asr"]
                def_acc, def_acc_std = metrics["defended_accuracy"]
                def_asr, def_asr_std = metrics["defended_asr"]
                time = metrics["time"]
                print(f"{dataset:<12} {attack:<16} {acc:.2f}±{acc_std:.2f}    {asr:.2f}±{asr_std:.2f}    "
                      f"{def_acc:.2f}±{def_acc_std:.2f}    {def_asr:.2f}±{def_asr_std:.2f}    {time:.2f}")
    else:
        print(f"{'Dataset':<12} {'Attack':<16} {'Accuracy':<15} {'ASR':<15} {'Time':<12}")
        print("-" * 100)
        for dataset, attacks in results.items():
            for attack, metrics in attacks.items():
                acc, acc_std = metrics["accuracy"]
                asr, asr_std = metrics["asr"]
                time = metrics["time"]
                print(f"{dataset:<12} {attack:<16} {acc:.2f}±{acc_std:.2f}    {asr:.2f}±{asr_std:.2f}    {time:.2f}")
    print("=" * 100)

if __name__ == "__main__":
    args = load_argument()
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    set_random_seed(args.seed)
    
    if args.dataset:
        datasets = [args.dataset.lower()]
    else:
        datasets = ["gtsrb", "cifar10", "cifar100", "svhn"]

    if args.attack_type:
        attack_configs = [(args.attack_type.lower(), args.trigger_type)]
        attack_names = [f"{args.attack_type.capitalize()}_{args.trigger_type or 'none'}"]
    else:
        attack_configs = [
            ("dcinject", "frequency"),
            ("dcinject", "adaptive_frequency"),
            ("Badpfl", None),
            ("Badnet", None)
        ]
        attack_names = ["DCInject_freq", "DCInject_adap_freq"]
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*50}\nRunning experiments for {dataset.upper()}\n{'='*50}")
        
        # Load data once per dataset
        train_dataset, test_dataset, train_labels, test_labels, num_classes = load_dataset(dataset)
        clients = setup_clients(args, train_dataset, test_dataset, train_labels, test_labels, num_classes, device)
        
        results[dataset] = {}
        
        # Run all attacks for this dataset
        for (attack_type, trigger_type), attack_name in zip(attack_configs, attack_names):
            print(f"\nRunning {attack_name} attack... {attack_type}  {trigger_type}")
            acc, acc_std, asr, asr_std, def_acc, def_acc_std, def_asr, def_asr_std, time_taken = run_experiment(
                args, attack_type, clients, num_classes, device, trigger_type)
            
            results[dataset][attack_name] = {
                "accuracy": (acc, acc_std),
                "asr": (asr, asr_std),
                "defended_accuracy": (def_acc, def_acc_std),
                "defended_asr": (def_asr, def_asr_std),
                "time": time_taken
            }
            print(f"Results: Acc={acc:.2f}±{acc_std:.2f}, ASR={asr:.2f}±{asr_std:.2f}, Time={time_taken:.2f}min")
            if args.defense and not args.no_defense:
                print(f"Defense: Acc={def_acc:.2f}±{def_acc_std:.2f}, ASR={def_asr:.2f}±{def_asr_std:.2f}")
    
    # Determine if defense is enabled
    defense_enabled = args.defense and not args.no_defense
    print_results(results, defense_enabled)