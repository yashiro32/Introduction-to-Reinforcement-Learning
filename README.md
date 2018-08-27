# Introduction to Reinforcement Learning

### Blackjack
There are 2 algorithm: Monte Carlo and Sarsa(?); the files are mc.py and sarsa_lambda.py. To train either one of them over 100000 episode for example run the following command

python mc.py 100000
or

python sarsa_lambda.py 100000
Once the training completes, the program will write out the Q-Values into the data directory. mc.py will produce a file call montecarlo_100000.pickle file; sarsa_lambda.py will produce a file called sarsa-lambda_100000.pickle.

To play Blackjack using the train Q-values run play.py file as below

python play.py montecarlo 100000
where the agent will use the Q-Values from montecarlo_100000.pickle file.