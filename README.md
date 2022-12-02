# ERGCN
Enhanced Relational Graph Convolution Network



## Required packages

Our model is written in Python3 and need install [PyTorch](https://pytorch.org/) and [DGL](https://www.dgl.ai) before starting.



## Before training

We collect 6 common datasets for entity prediction tasks, they are in the document **data**. Preprocessing datasets(i.e. WIKI) before training via the following code:

~~~
python get_data_history.py -d WIKI
~~~

Then, training the global model:

~~~
python pretrain.py -d WIKI --dropout 0.2 --h_dim 200 --max_epochs 250 --seq_len 5 --lr 0.001 --batch_size 96 --gpu 0
~~~

-d ***dataset_name***

--dropout ***dropout_rate***

--h_dim ***the length of embedding***

--max_epochs ***Iterations***

--seq_len ***the depth of historical information used in the model***

--lr ***learning rate***

--batch_size ***batch size***

--gpu ***number of GPU***



## Start Training 

Then, training the ERGCN model:

~~~
python train.py -d WIKI --dropout 0.2 --h_dim 200 --max_epochs 20 --seq_len 5 --lr 0.002 --batch_size 1024 --reverse False --alpha 0.1 --layers 2 --name 1 
~~~

notation of the new commands:

--reverse :Means the direction of training. If **FALSE**, the model uses the **subjects** to predict the **objects** in the Temporal Knowledge Graphs, vice versa.

--alpha :Means the balance weight before loss1 and loss2

--layers :Means the rippled size of nieghbor nodes around the source node. If the *--layers* is less than the layer number of ERGCN block used in the model, model will not update new features of the node from its neighbor nodes in the aggregator block.

--name :Used for saving different model types(use *int*, not *str*).

## Testing 

After training, we can start testing:

~~~
python test.py -d WIKI --filter 0 --name 1
~~~

notation of the new commands:

--filter :metric for testing. If **zero**, print the **raw results**. Else, print the results with ground truth.

Notice that the