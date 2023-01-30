# Data-Free Knowledge Distillation for Heterogeneous Federated Learning

Research code that accompanies the paper [Data-Free Knowledge Distillation for Heterogeneous Federated](https://arxiv.org/pdf/2105.10056.pdf).
It contains implementation of the following algorithms:
* **FedDistill** and its extension **FedDistll-FL** ([paper](https://arxiv.org/pdf/2011.02367.pdf) and [code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverFedDistill.py)).

## Install Requirements:
```pip3 install -r requirements.txt```

  
## Prepare Dataset: 
* To generate *non-iid* **Mnist** Dataset following the Dirichlet distribution D(&alpha;=0.1) for 20 clients, using 50% of the total available training samples:
<pre><code>cd FedGen/data/Mnist
python generate_niid_dirichlet.py --n_class 10 --sampling_ratio 0.5 --alpha 0.1 --n_user 20
### This will generate a dataset located at FedGen/data/Mnist/u20c10-alpha0.1-ratio0.5/
</code></pre>


## Run Experiments: 

There is a main file "main.py" which allows running all experiments.

#### Run experiments on the *Mnist* Dataset:
```
python main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm FedGen --batch_size 32 --num_glob_iters 200 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 3 
python main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm FedAvg --batch_size 32 --num_glob_iters 200 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 3 
python main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm FedProx --batch_size 32 --num_glob_iters 200 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 3 
python main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm FedDistll-FL --batch_size 32 --num_glob_iters 200 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 3 
```
----