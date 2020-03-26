# Double Deep Q-network
This is a modified version of DDQN (Double Deep Q-network) of listed references based on Tensorflow. 

### Prerequisites
- Python 3 or above
- tensorflow 1.14
- gym 0.9.2

### performance
- CartPole-v0

It takes about 10 minutes to run 1e5 steps on computer with 2 GPU 2080 Ti, Intel i9-9820X CPU (10 Cores) and 128 GB Memory. The algorithm takes about 2 minutes to converge.

<p align="center">
     <img src="docs/ddqn_er.png" alt="output_example" width="60%" height="60%">
     <br>Fig.1 Training process on CartPole-v0
</p>

<p align="center">
     <img src="docs/ddqn_tr.png" alt="output_example" width="60%" height="60%">
     <br>Fig.2 Evaluation process on CartPole-v0
</p>

You can also monitor the training process by type
`tensorboard --logdir=Data/log/ --host localhost --port 8088` (on windows). or `tensorboard --logdir=Data/log/` (on Linux).

<p align="center">
     <img src="docs/CartPole_tb.png" alt="output_example" width="60%" height="60%">
     <br>Fig.3 Tensorboard plot of CartPole-v0
</p>

### References
[1. OpenAI Baselines](https://github.com/openai/baselines)

[2. cts198859's Implementation](https://github.com/cts198859/deeprl_signal_control)


