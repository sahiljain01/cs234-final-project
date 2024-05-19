## CS 234 Final Project

### Project Goal

Our project aims to explore the application of reinforcement learning, implemented through
the FinRL library, for stock price prediction. Specifically, we first seek to replicate the results
that the FinRL paper shares on state-of-the-art RL algorithms; then, we seek to evaluate those
results with our validation methods, including purged K-fold cross-validation to introduce
more robust checks for overfitting. Finally, we plan to explore a couple of directions to
improve upon their results, including enhancing the state space (such as using a GNN to
create a vector representation of the environment leveraging message passing) and
implementing RL algorithms that the FinRL paper has not already leveraged. Ultimately, we
hope our work can serve as an open-source contribution to the FinRL library for other
researchers to leverage in their work too.

### Environment set up

To set up the environment for this assignment, you will need to create a new
`conda` environment.

    conda create -n cs234_final_project python=3.8

Once you activate it, first run:

    brew install cmake openmpi 
    brew install swig 

Then, run:

    pip install -r requirements.txt

After this you are good to go!
