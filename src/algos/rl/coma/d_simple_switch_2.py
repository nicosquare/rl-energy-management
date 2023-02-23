import torch
import gym
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from src.utils.tools import set_all_seeds

class Memory:
    def __init__(self, agent_num, action_dim):
        self.agent_num = agent_num
        self.action_dim = action_dim

        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(agent_num)]
        self.reward = []
        self.done = [[] for _ in range(agent_num)]

    def get(self):
        actions = torch.tensor(self.actions)
        observations = self.observations

        pi = []
        for i in range(self.agent_num):
            pi.append(torch.cat(self.pi[i]).view(len(self.pi[i]), self.action_dim))

        reward = torch.tensor(self.reward)
        done = self.done

        return actions, observations, pi, reward, done

    def clear(self):
        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(self.agent_num)]
        self.reward = []
        self.done = [[] for _ in range(self.agent_num)]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class RNNActor(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNActor, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args['rnn_hidden_dim'])
        self.rnn = nn.GRUCell(args['rnn_hidden_dim'], args['rnn_hidden_dim'])
        self.fc2 = nn.Linear(args['rnn_hidden_dim'], args['n_actions'])

        self.hidden_state = self.init_hidden()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args['rnn_hidden_dim']).zero_()[0]

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        # h_in = self.hidden_state.reshape(-1, self.args['rnn_hidden_dim'])
        h_in = self.hidden_state
        h = self.rnn(x, h_in)

        self.hidden_state = h.detach().clone()

        q = F.softmax(self.fc2(h))

        return q, h

class Critic(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim):
        super(Critic, self).__init__()

        input_dim = 1 + state_dim * agent_num + agent_num

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class COMA:
    def __init__(self, agent_num, state_dim, action_dim, lr_c, lr_a, gamma, target_update_steps):
        self.agent_num = agent_num
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma

        self.target_update_steps = target_update_steps

        self.memory = Memory(agent_num, action_dim)

        # self.actors = [Actor(state_dim, action_dim) for _ in range(agent_num)]
        self.actors = [RNNActor(input_shape=state_dim, args={
            'rnn_hidden_dim': 64,
            'n_actions': action_dim
        }) for _ in range(agent_num)]
        self.critic = Critic(agent_num, state_dim, action_dim)

        self.critic_target = Critic(agent_num, state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=lr_a) for i in range(agent_num)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        self.count = 0

    def get_actions(self, observations):
        observations = torch.tensor(observations)

        actions = []

        for i in range(self.agent_num):
            
            dist, _ = self.actors[i](observations[i])
            action = Categorical(dist).sample()

            self.memory.pi[i].append(dist)
            actions.append(action.item())

        self.memory.observations.append(observations)
        self.memory.actions.append(actions)

        return actions

    def train(self):
        actor_optimizer = self.actors_optimizer
        critic_optimizer = self.critic_optimizer

        actions, observations, pi, reward, done = self.memory.get()

        for i in range(self.agent_num):
            # train actor

            input_critic = self.build_input_critic(i, observations, actions)
            Q_target = self.critic_target(input_critic).detach()

            action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)

            baseline = torch.sum(pi[i] * Q_target, dim=1).detach()
            Q_taken_target = torch.gather(Q_target, dim=1, index=action_taken).squeeze()
            advantage = Q_taken_target - baseline

            log_pi = torch.log(torch.gather(pi[i], dim=1, index=action_taken).squeeze())

            actor_loss = - torch.mean(advantage * log_pi)

            actor_optimizer[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 5)
            actor_optimizer[i].step()

            # train critic

            Q = self.critic(input_critic)

            action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)
            Q_taken = torch.gather(Q, dim=1, index=action_taken).squeeze()

            # TD(0)
            r = torch.zeros(len(reward[:, i]))
            for t in range(len(reward[:, i])):
                if done[i][t]:
                    r[t] = reward[:, i][t]
                else:
                    r[t] = reward[:, i][t] + self.gamma * Q_taken_target[t + 1]

            critic_loss = torch.mean((r - Q_taken) ** 2)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
            critic_optimizer.step()

        if self.count == self.target_update_steps:
            self.critic_target.load_state_dict(agents.critic.state_dict())
            self.count = 0
        else:
            self.count += 1

        self.memory.clear()

    def build_input_critic(self, agent_id, observations, actions):
        batch_size = len(observations)

        ids = (torch.ones(batch_size) * agent_id).view(-1, 1)

        observations = torch.cat(observations).view(batch_size, self.state_dim * self.agent_num)
        input_critic = torch.cat([observations.type(torch.float32), actions.type(torch.float32)], dim=-1)
        input_critic = torch.cat([ids, input_critic], dim=-1)

        return input_critic

if __name__ == "__main__":

    set_all_seeds(0)

    # Hyperparameters
    agent_num = 2

    state_dim = 2
    action_dim = 5

    gamma = 0.99
    lr_a = 0.00006
    lr_c = 0.0005

    target_update_steps = 10

    # agent initialisation

    agents = COMA(agent_num, state_dim, action_dim, lr_c, lr_a, gamma, target_update_steps)

    env = gym.make("ma_gym:Switch2-v0")
    obs = env.reset()
    
    for agent in agents.actors:
        agent.init_hidden()

    episode_reward = 0
    episodes_reward = []

    # training loop

    n_episodes = 20000
    episode = 0

    while episode < n_episodes:
        actions = agents.get_actions(obs)

        # env.render()
        next_obs, reward, done_n, _ = env.step(actions)

        agents.memory.reward.append(reward)
        for i in range(agent_num):
            agents.memory.done[i].append(done_n[i])

        episode_reward += sum(reward)

        obs = next_obs

        if all(done_n):
            episodes_reward.append(episode_reward)
            episode_reward = 0

            episode += 1

            obs = env.reset()

            if episode % 10 == 0:

                for agent in agents.actors:
                    agent.init_hidden()

                agents.train()

            if episode % 100 == 0:
                print(f"episode: {episode}, average reward: {sum(episodes_reward[-100:]) / 100}")

    print('Orale')