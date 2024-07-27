#pragma once

#include <torch/torch.h>
#include <random>

#include "Models.h"

using uint = unsigned int;

// Vector of tensors.
using VT = std::vector<torch::Tensor>;

// Optimizer.
using OPT = torch::optim::Optimizer;

// Random engine for shuffling memory.
std::random_device rd;
std::mt19937 re(rd());

// Proximal policy optimization, https://arxiv.org/abs/1707.06347
class PPO
{
public:
    static auto returns(VT& rewards, VT& dones, VT& vals, double gamma, double lambda) -> VT; // Generalized advantage estimate, https://arxiv.org/abs/1506.02438
    static auto update(ActorCritic& ac,
	    std::shared_ptr<torch::optim::Adam> &opt, 
                       uint steps, uint epochs, uint mini_batch_size, double beta, float gamma, c10::DeviceType device, double clip_param = .2) -> void;
    static auto save_replay(torch::Tensor &states,
	    torch::Tensor &actions,
	    torch::Tensor &log_probs,
	    torch::Tensor &returns,
	    std::vector<bool> &dones,
	    torch::Tensor &advantages) -> void;
    static auto count_of_replays() -> size_t;
};

// Replay buffer for experience replay
class ReplayBuffer
{
public:
	ReplayBuffer(size_t capacity) :
		capacity(capacity) {}

	void add(const torch::Tensor &state, const torch::Tensor &action, const torch::Tensor &log_prob, const torch::Tensor &reward, const std::vector<bool> &dones, const torch::Tensor &advantage)
	{
		if(buffer.size() == capacity)
		{
			buffer.erase(buffer.begin());
		}
		buffer.push_back({state, action, log_prob, reward, dones, advantage});
	}

    void clear()
    {
	    buffer.clear();
    }

    size_t size()
    {
	    return buffer.size();
    }

	std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<bool>, torch::Tensor>> sample(size_t batch_size)
	{
		std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<bool>, torch::Tensor>> batch;
		std::sample(buffer.begin(), buffer.end(), std::back_inserter(batch), batch_size, std::mt19937{std::random_device{}()});
		return batch;
	}

private:
	size_t capacity;
	std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<bool>, torch::Tensor>> buffer;
};

torch::Tensor normalize_rewards(const torch::Tensor &rewards)
{
	auto mean = rewards.mean();
	auto std = rewards.std();
	return (rewards - mean) / (std + 1e-8);
}

static ReplayBuffer replay_buffer(128000);

auto PPO::returns(VT& rewards, VT& dones, VT& vals, double gamma, double lambda) -> VT
{
    // Compute the returns.
	torch::Tensor gae = torch::zeros({1}, torch::kF32);
    VT returns(rewards.size(), torch::zeros({1}, torch::kF32));

    for (uint i=rewards.size();i-- >0;) // inverse for loops over unsigned: https://stackoverflow.com/questions/665745/whats-the-best-way-to-do-a-reverse-for-loop-with-an-unsigned-index/665773
    {
        // Advantage.
        auto delta = rewards[i] + gamma*vals[i+1]*(1-dones[i]) - vals[i];
        gae = delta + gamma*lambda*(1-dones[i])*gae;

        returns[i] = gae + vals[i];
    }

    return returns;
}

torch::Tensor compute_advantages(ActorCritic &ac, const torch::Tensor &returns, const torch::Tensor &states)
{
	auto [policy_actions, values] = ac->forward(states);
	return returns - values.squeeze(-1);
}

torch::Tensor calculate_returns(const torch::Tensor &rewards, const std::vector<bool> &dones, float gamma)
{
	//printf("FINNNN1.1\n");
	torch::Tensor returns = torch::zeros({rewards.size(0), 1}, rewards.options());
	//printf("FINNNN1.2\n");
	torch::Tensor G = torch::zeros({1}, rewards.options());
	//printf("FINNNN1.3\n");
	//std::cout << returns.sizes() << std::endl;
	//std::cout << G.sizes() << std::endl;
	for(int64_t i = rewards.size(0) - 1; i >= 0; --i)
	{
		G = rewards[i] + gamma * G * dones[i];
		//printf("FINNNN1.4\n");
		//G = rewards[i] + gamma * G;
		//std::cout << G.item<float>() << std::endl;
		//printf("FINNNN1.5\n");
		returns[i] = G;
		//printf("FINNNN1.6\n");
	}

	//printf("FINE\n");
	//std::cout << rewards.sizes() << std::endl;
	//std::cout << returns.sizes() << std::endl;

	return returns.detach();
}

auto PPO::save_replay(torch::Tensor& state,
    torch::Tensor& action,
    torch::Tensor& log_prob,
    torch::Tensor& returns,
	std::vector<bool>& dones,
    torch::Tensor& advantage) -> void
{
	replay_buffer.add(state, action, log_prob, returns, dones, advantage);
}

auto PPO::count_of_replays() -> size_t
{
	return replay_buffer.size();
}

auto PPO::update(ActorCritic& ac,
				std::shared_ptr<torch::optim::Adam>& opt, 
                 uint steps, uint epochs, uint mini_batch_size, double beta, float gamma, c10::DeviceType device, double clip_param) -> void
{
	double total_loss = 0.0;

    for (uint e=0;e<epochs;e++)
    {
        // Generate random indices.
        /*torch::Tensor cpy_sta = torch::zeros({mini_batch_size, states.size(1)}, states.options());
        torch::Tensor cpy_act = torch::zeros({mini_batch_size, actions.size(1)}, actions.options());
        torch::Tensor cpy_log = torch::zeros({mini_batch_size, log_probs.size(1)}, log_probs.options());
        torch::Tensor cpy_ret = torch::zeros({mini_batch_size, returns.size(1)}, returns.options());
        torch::Tensor cpy_adv = torch::zeros({mini_batch_size, advantages.size(1)}, advantages.options());*/
		//printf("UPDATING0\n");
        auto batch = replay_buffer.sample(mini_batch_size / 64);
		std::vector<torch::Tensor> states, actions, log_probs, rewards;
		std::vector<bool> dones;
	    for(const auto &[state, action, log_prob, reward, done, advantage] : batch)
	    {
		    states.push_back(state);
		    actions.push_back(action);
		    log_probs.push_back(log_prob);
			rewards.push_back(reward);
		    dones.insert(dones.end(), done.begin(), done.end());
		    //advantages.push_back(advantage);
		    //std::cout << log_prob.sizes() << std::endl;
	    }
	    //printf("UPDATING0.1\n");
	    torch::Tensor cpy_sta = torch::cat(states).detach();
	    //std::cout << cpy_sta.sizes() << std::endl;
	    //printf("UPDATING0.2\n");
	    torch::Tensor cpy_inputs = cpy_sta.index({"...", torch::indexing::Slice(0, 9)});
	    //printf("UPDATING0.3\n");
	    torch::Tensor cpy_blocks = torch::one_hot(cpy_sta.index({"...", torch::indexing::Slice(9, 1097)}).to(torch::kInt64), 4).to(torch::kF32).view({mini_batch_size, -1});
	    //printf("UPDATING0.4\n");
	    cpy_sta = torch::cat({cpy_inputs, cpy_blocks}, 1);
	    //std::cout << cpy_sta.sizes() << std::endl;
	    //printf("UPDATING0.1.1\n");
	    torch::Tensor cpy_act = torch::cat(actions).detach();
	    //printf("UPDATING0.1.2\n");
	    torch::Tensor cpy_log = torch::cat(log_probs).detach().to(device);
	    //printf("UPDATING0.1.3\n");
	    auto catted = torch::cat(rewards);
	    //printf("UPDATING0.1.3.1\n");
	    //std::cout << catted.sizes() << std::endl;
	    //printf("UPDATING0.1.3.2\n");
	    torch::Tensor cpy_ret = normalize_rewards(calculate_returns(catted, dones, gamma)).detach();
	    //printf("UPDATING0.1.4\n");
	    torch::Tensor cpy_adv = compute_advantages(ac, cpy_ret, cpy_sta.view({mini_batch_size, 1, 4361})).detach().to(device);
	    //std::cout << cpy_ret << std::endl;

        /*for (uint b=0;b<mini_batch_size;b++) {

            uint idx = std::uniform_int_distribution<uint>(0, steps-1)(re);
            cpy_sta[b] = states[idx];
            cpy_act[b] = actions[idx];
            cpy_log[b] = log_probs[idx];
            cpy_ret[b] = returns[idx];
            cpy_adv[b] = advantages[idx];
        }*/
	    //printf("UPDATING1.1\n");
        auto av = ac->forward(cpy_sta); // action value pairs
	    //printf("UPDATING1.2\n");
        auto action = std::get<0>(av);
        auto entropy = ac->entropy().mean();
        auto new_log_prob = ac->log_prob(cpy_act);
	    //printf("UPDATING1.3\n");
        auto old_log_prob = cpy_log;
	    //printf("UPDATING1.3.1\n");
		//std::cout << new_log_prob.sizes() << " " << old_log_prob.sizes() << std::endl;
        auto ratio = (new_log_prob - old_log_prob).exp();
	    //printf("UPDATING1.4\n");
		//std::cout << ratio.sizes() << std::endl;
	    //std::cout << cpy_adv.sizes() << std::endl;
        auto surr1 = ratio * cpy_adv;
	    //printf("UPDATING1.4.1\n");
        auto surr2 = torch::clamp(ratio, 1. - clip_param, 1. + clip_param) * cpy_adv;
	    //printf("UPDATING1.5\n");
        auto val = std::get<1>(av);
	    //printf("UPDATING1.6\n");
        auto actor_loss = -torch::min(surr1, surr2).mean();
	    //printf("UPDATING1.7\n");
	    auto critic_loss = torch::nn::functional::mse_loss(val, cpy_ret); //(cpy_ret - val).pow(2).mean();
	    //printf("UPDATING1.8\n");
        auto loss = 0.5 * critic_loss + actor_loss - beta * entropy;
	    //printf("UPDATING1.9\n");
        //std::cout << "Actor Loss: " << actor_loss.item<double>() << ", Critic Loss: " << critic_loss.item<double>() << std::endl;

        opt->zero_grad();
	    //printf("UPDATING1.10\n");
        loss.backward();
	    //printf("UPDATING1.11\n");
        opt->step();
	    //printf("UPDATING1.12\n");
		total_loss += loss.item<double>();
    }

	double avg_loss = total_loss / epochs;
    std::cout << "Average training Loss: " << avg_loss << std::endl;

    replay_buffer.clear();
}
