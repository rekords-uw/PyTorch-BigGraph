### Randomness Experiments ###

View the current commit to see what exactly I changed in the code to allow for randomness changes.

#### Where to modify the code ####

model.py
* SimpleEmbedding(AbstractEmbedding): The constructor sets up the list for pulling indices. sample_entities() 
is what is called when negative samples need to be produced.
* prepare_negatives() calls sample_entities() to produce the negative samples.
* set_embeddings() initializes the SimpleEmbedding() 

train.py
* train_and_report_stats() is the method that does the training and spawns training workers. 


config.py
* Add whatever configuration flags you want here. I added shuffled_mode, shuffle_size, and shuffle_order here so I could pass these in as command line arguments.

examples/fb15k.py
* Example code that runs on a subset of the freebase data set. Modified to support shuffle_mode, shuffle_size, and shuffle_order

examples/configs/fb15k_config.py
* Here you can modify the default configuration for training. Fb15k converges at around 5 epochs, so keep that in mind here. eval_fraction holds out a fraction of the training set for validation. 

#### Installation/Cloudlab deployment ####
Setup a ubuntu18 cloublab node and clone this repository. When the node is up, run the following commands.

* `sudo apt update`
* `sudo apt-get install python3`
* `sudo apt-get install python3-pip`

Go into the base directory of the repo and run `pip3 install .`

This will place install the relevant packages and torchbiggraph code in your `~/.local/lib/python3.6` directory. 

Note that if you modify the code from the cloned repo, you will have to make sure pip3 install again to have your changes take effect.

#### Running Instructions: Single Experiment ####
A single experiment can be run by going to the base directory and running

`python3 torchbiggraph/examples/fb15k.py --shuffle_mode all --shuffle_size 1`

This will download the relevant dataset and place in the `data/` directory in the directory where you ran the code.

After the data is downloaded and formatted the training process will begin and will create a directory called `model/` in your current directory. 

`model/` contains checkpointed embeddings and training stats in json format. Copy these training stats to save them for plotting/analysis.


Finally once training is complete, evaluation will be performed. You might notice the evaluation MRR will be much higher than the mrr reported during training. This is because the evaluation uses filteredMRR rather than regular MRR.

When running another experiment make sure to clear the `model/` directory and save your training stats somewhere. Deleting `data/` is not necessary between runs. 

#### Running Instructions: Many Experiments ####
Create a shell script that passes different arguments to the previous command. Be sure to use nohup to disown the process in order to have it continue running while you close your terminal session.

`nohup <your_script>.sh &`

This will put all stdout in nohup.out which you can monitor with `tail -f nohup.txt`

Between each run copy the training stats. I changed the names of the training stats file to account for the test case that was run. Ex: `training_stats_<randomness_mode>_<list_size>_<run_number>.json`
#### Plotting Instructions ####

`experiments/plotting/plotting.py` can parse the json files produced by the training. Look through this code and edit the relevant data paths / parsing methods to suit your needs. It's a little rough in here but it plots means/std of training loss, training mrr and evaluation mrr.

#### General things to do ####
* Explore how the different randomness configurations effects variance during training. 
* Create synthetic graph data sets to see how the different configurations effect convergence and embedding quality for various graphical structures. Some data sets could be chains, fully connected graphs, sparse graphs, graphs with cliques, etc...
* Think about how we can increase randomness without increasing list size. 

#### Specific Tasks ####
1. Run tests on fb15k with shuffle_mode = all and shuffle_size = [1, 2, 4, ... 64] and see how the standard deviation changes with increased shuffle_size. Compare these results with the variance of shuffle_mode = uniform. I previously did runs of 5 for each test case but some cases had pretty bad outliers. You could probably up that to around 15 or 20 to sufficiently get a good estimate of std. 
2. Generate two data sets, a chain, and a fully connected graph. Run the various randomness configurations on both and compare results. To see how PBG likes it's files formatted, checkout the tsvs for the fb15k data set.