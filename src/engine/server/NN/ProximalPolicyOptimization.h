#pragma once

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <random>

#include "Models.h"
#include <Windows.h>

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
	static auto Initilize(size_t batch_size, size_t count_players) -> void;
    static auto update(ActorCritic& ac,
	    std::shared_ptr<torch::optim::Adam> &opt, 
                       uint steps, uint epochs, uint mini_batch_size, double beta, float gamma, c10::DeviceType device, double clip_param = .2) -> double;
    static auto save_replay(torch::Tensor &states,
	    torch::Tensor &actions,
	    torch::Tensor &log_probs,
	    torch::Tensor &returns,
	    std::vector<bool> &dones) -> void;
    static auto count_of_replays() -> size_t;
};

// Replay buffer for experience replay
class ReplayBuffer
{
public:
	ReplayBuffer(size_t capacity, size_t count_players) :
		capacity(capacity), count_players(count_players)
	{
		dones.resize(capacity / count_players);
	}

	void add(const torch::Tensor &state, const torch::Tensor &action, const torch::Tensor &log_prob, const torch::Tensor &reward, const std::vector<bool> &dones)
	{
		//torch::NoGradGuard no_grad;
		/*if(buffer.size() == capacity)
		{
			buffer.erase(buffer.begin());
		}*/
		//buffer.push_back({state, action, log_prob, reward, dones, advantage});

		/*for(size_t i = 0; i < dones.size(); i++)
		{
			std::get<4>(buffer)[i].push_back(dones[i]);
		}*/

		/*std::cout << state.sizes() << std::endl;
		std::cout << action.sizes() << std::endl;
		std::cout << log_prob.sizes() << std::endl;
		std::cout << reward.sizes() << std::endl;
		std::cout << advantage.sizes() << std::endl;*/

		states.push_back(state.unsqueeze(1));
		actions.push_back(action.unsqueeze(1));
		log_probs.push_back(log_prob.unsqueeze(1));
		rewards.push_back(reward);
		//advantages.push_back(state.unsqueeze(1));

		//if(actions.size(0) == 0)
		//{
		//	//states = state.reshape({64, 1, 1104});
		//	actions = action.reshape({64, 1, 7});
		//	log_probs = log_prob.reshape({64, 1, 7});
		//	rewards = reward;
		//	advantages = advantage;
		//}
		//else
		//{
		//	//states = torch::cat({states, state.unsqueeze(1)}, 1);
		//	//std::cout << c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes[0].allocated << std::endl;
		//	//std::cout << states.numel() * states.element_size() << std::endl;
		//	//std::cout << states.sizes() << std::endl;
		//	actions = torch::cat({actions, action.unsqueeze(1)}, 1);
		//	log_probs = torch::cat({log_probs, log_prob.unsqueeze(1)}, 1);
		//	rewards = torch::cat({rewards, reward}, 1);
		//	advantages = torch::cat({advantages, advantage}, 1);
		//}

		for(size_t i = 0; i < count_players; i++)
		{
			this->dones[i].push_back(dones[i]);
		}

		/*std::cout << "States size: " << states.sizes() << std::endl;
		std::cout << "Actions size: " << actions.sizes() << std::endl;
		std::cout << "Log_probs size: " << log_probs.sizes() << std::endl;
		std::cout << "Rewards size: " << rewards.sizes() << std::endl;
		std::cout << "Advantages size: " << advantages.sizes() << std::endl;*/
	}

    void clear()
    {
	    // Clear the tensors by reinitializing them to empty
	    states.clear();
	    actions.clear();
	    log_probs.clear();
	    rewards.clear();

		states_concatenated = torch::Tensor();
	    actions_concat = torch::Tensor();
		log_probs_concat = torch::Tensor();
	    rewards_concat = torch::Tensor();
		dones_concat.clear();
	    //advantages.clear();

		for(size_t i = 0; i < count_players; i++)
	    {
		    this->dones[i].clear();
	    }
    }

    size_t size()
    {
	    return dones.size();
    }

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<bool>> sample(size_t batch_size)
	{
		//std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<bool>, torch::Tensor>> batch;
		//printf("1\n");
		//printf("1\n");
		//printf("1\n");

		if(dones_concat.size() == 0)
		{
			// printf("creating\n");
			// Sleep(3000);
			states_concatenated = torch::cat(states, 1);
			// printf("states catted\n");
			// Sleep(7000);
			// states_concatenated = torch::Tensor();
			// printf("states deleted\n");
			// Sleep(7000);
			//  printf("1\n");
			actions_concat = torch::cat(actions, 1);
			// printf("1\n");
			log_probs_concat = torch::cat(log_probs, 1);
			// printf("1\n");
			rewards_concat = torch::cat(rewards, 1);
			// printf("created\n");
			//  advantages_concat = torch::cat(rewards, 1);
			//  printf("1\n");

			states_concatenated = states_concatenated.reshape({states_concatenated.sizes()[0] * states_concatenated.sizes()[1], states_concatenated.sizes()[2]});
			actions_concat = actions_concat.reshape({actions_concat.sizes()[0] * actions_concat.sizes()[1], actions_concat.sizes()[2]});
			// printf("1\n");
			log_probs_concat = log_probs_concat.reshape({log_probs_concat.sizes()[0] * log_probs_concat.sizes()[1], log_probs_concat.sizes()[2]});
			// printf("1\n");
			// std::cout << "Rewards size: " << rewards.sizes() << " " << rewards.size(0) << std::endl;
			rewards_concat = rewards_concat.reshape({rewards_concat.numel(), 1});
			// printf("1\n");
			// advantages_concat = advantages_concat.reshape({advantages_concat.numel()});
			// printf("1\n");
			for(size_t i = 0; i < count_players; i++)
			{
				dones_concat.insert(dones_concat.end(), dones[i].begin(), dones[i].end());
			}
		}

		//printf("1\n");
		std::vector<bool> dones_concat_ret;
		torch::Tensor states_concatenated_ret, actions_concat_ret, log_probs_concat_ret, rewards_concat_ret;

		size_t start = round(((float)std::rand() / (float)RAND_MAX) * (float)(dones_concat.size() - batch_size - 1));
		size_t end = start + batch_size;

		for(size_t i = end - 1; i > start; i--)
		{
			if(dones_concat[i])
			{
				end = i + 1;
				break;
			}
		}

		states_concatenated_ret = states_concatenated.index({torch::indexing::Slice(start, end)});
		actions_concat_ret = actions_concat.index({torch::indexing::Slice(start, end)});
		log_probs_concat_ret = log_probs_concat.index({torch::indexing::Slice(start, end)});
		rewards_concat_ret = rewards_concat.index({torch::indexing::Slice(start, end)});
		//advantages_concat = advantages_concat.index({torch::indexing::Slice(start, end)});
		dones_concat_ret = std::vector<bool>(dones_concat.begin() + start, dones_concat.begin() + end);
		/*std::cout << "States size: " << states_concatenated.sizes() << std::endl;
		std::cout << "Actions size: " << actions_concat.sizes() << std::endl;
		std::cout << "Log_probs size: " << log_probs_concat.sizes() << std::endl;
		std::cout << "Rewards size: " << rewards_concat.sizes() << std::endl;
		std::cout << "Advantages size: " << advantages_concat.sizes() << std::endl;
		std::cout << "Dones size: " << dones_concat.size() << std::endl;*/
		//printf("ereer\n");
		//std::cout << states.sizes() << std::endl;
		//std::sample(buffer.begin(), buffer.end(), std::back_inserter(batch), batch_size, std::mt19937{std::random_device{}()});
		return {states_concatenated_ret, actions_concat_ret, log_probs_concat_ret, rewards_concat_ret, dones_concat_ret};
	}

private:
	size_t count_players;
	size_t capacity;
	torch::Tensor states_concatenated, actions_concat, log_probs_concat, rewards_concat;
	std::vector<bool> dones_concat;
	std::vector<torch::Tensor> states, actions, log_probs, rewards;
	//std::tuple < torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<std::vector<bool>>, torch::Tensor> buffer;
	//torch::Tensor actions, log_probs, rewards, advantages;
	std::vector<std::vector<bool>> dones;
	//std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<bool>, torch::Tensor>> buffer;
};

torch::Tensor normalize_rewards(const torch::Tensor &rewards)
{
	auto mean = rewards.mean();
	auto std = rewards.std();
	return (rewards - mean) / (std + 1e-8);
}

static ReplayBuffer* replay_buffer = nullptr;

auto PPO::Initilize(size_t batch_size, size_t count_players) -> void
{
	replay_buffer = new ReplayBuffer(batch_size, count_players); // (256000, 256)
}

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
		G = rewards[i] + gamma * G * (!dones[i]);
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
	std::vector<bool>& dones) -> void
{
	//torch::NoGradGuard no_grad;
	replay_buffer->add(state, action, log_prob, returns, dones);
}

auto PPO::count_of_replays() -> size_t
{
	return replay_buffer->size();
}

auto PPO::update(ActorCritic &ac,
	std::shared_ptr<torch::optim::Adam> &opt,
	uint steps, uint epochs, uint mini_batch_size, double beta, float gamma, c10::DeviceType device, double clip_param) -> double
{
	double total_loss = 0.0;

	{
		std::deque<torch::Tensor> states, actions, log_probs, rewards;
		std::deque<std::vector<bool>> dones;

		for(size_t i = 0; i < epochs; i++)
		{
			auto [state, action, log_prob, reward, done] = replay_buffer->sample(mini_batch_size);
			states.push_back(state);
			actions.push_back(action);
			log_probs.push_back(log_prob);
			rewards.push_back(reward);
			dones.push_back(done);
		}
		// printf("CHECK\n");
		// Sleep(10000);
		// printf("CLEARING\n");
		replay_buffer->clear();
		// torch::cuda::synchronize();
		// c10::cuda::CUDACachingAllocator::emptyCache();
		// Sleep(10000);

		for(uint e = 0; e < epochs; e++)
		{
			c10::cuda::CUDACachingAllocator::emptyCache();
			torch::Tensor states_cpy = states[0];
			torch::Tensor actions_cpy = actions[0];
			torch::Tensor log_probs_cpy = log_probs[0];
			torch::Tensor rewards_cpy = rewards[0];
			std::vector<bool> dones_cpy = dones[0];
			// Generate random indices.
			/*torch::Tensor cpy_sta = torch::zeros({mini_batch_size, states.size(1)}, states.options());
			torch::Tensor cpy_act = torch::zeros({mini_batch_size, actions.size(1)}, actions.options());
			torch::Tensor cpy_log = torch::zeros({mini_batch_size, log_probs.size(1)}, log_probs.options());
			torch::Tensor cpy_ret = torch::zeros({mini_batch_size, returns.size(1)}, returns.options());
			torch::Tensor cpy_adv = torch::zeros({mini_batch_size, advantages.size(1)}, advantages.options());*/
			// printf("UPDATING0\n");
			// auto [states, actions, log_probs, rewards, dones] = replay_buffer->sample(mini_batch_size);
			// std::vector<torch::Tensor> states, actions, log_probs, rewards;
			// std::vector<bool> dones;
			//   for(const auto &[state, action, log_prob, reward, done, advantage] : batch)
			//   {
			//    states.push_back(state);
			//    actions.push_back(action);
			//    log_probs.push_back(log_prob);
			//	rewards.push_back(reward);
			//    dones.insert(dones.end(), done.begin(), done.end());
			//    //advantages.push_back(advantage);
			//    //std::cout << log_prob.sizes() << std::endl;
			//   }
			// printf("UPDATING0.1\n");
			torch::Tensor cpy_sta = states_cpy;
			// std::cout << cpy_sta.sizes() << std::endl;
			// printf("UPDATING0.2\n");
			torch::Tensor cpy_inputs = cpy_sta.index({"...", torch::indexing::Slice(0, 16)});
			// printf("UPDATING0.3\n");
			torch::Tensor cpy_blocks = torch::one_hot(cpy_sta.index({"...", torch::indexing::Slice(16, 1105)}).to(torch::kInt64), 4).to(torch::kF32).view({cpy_sta.size(0), -1});
			// printf("UPDATING0.4\n");
			cpy_sta = torch::cat({cpy_inputs, cpy_blocks}, 1);
			// std::cout << cpy_sta.sizes() << std::endl;
			// printf("UPDATING0.1.1\n");
			torch::Tensor cpy_act = actions_cpy;
			// printf("UPDATING0.1.2\n");
			torch::Tensor cpy_log = log_probs_cpy;
			// printf("UPDATING0.1.3\n");
			// auto catted = torch::cat(rewards).reshape({mini_batch_size, 1});
			// printf("UPDATING0.1.3.1\n");
			// std::cout << catted.sizes() << std::endl;
			// printf("UPDATING0.1.3.2\n");
			auto returnsee = calculate_returns(rewards_cpy, dones_cpy, gamma);
			torch::Tensor cpy_ret = normalize_rewards(returnsee);
			// printf("UPDATING0.1.4\n");
			torch::Tensor cpy_adv = compute_advantages(ac, cpy_ret, cpy_sta.view({cpy_sta.size(0), 1, 4372}));
			// std::cout << cpy_ret << std::endl;

			/*for (uint b=0;b<mini_batch_size;b++) {

			    uint idx = std::uniform_int_distribution<uint>(0, steps-1)(re);
			    cpy_sta[b] = states[idx];
			    cpy_act[b] = actions[idx];
			    cpy_log[b] = log_probs[idx];
			    cpy_ret[b] = returns[idx];
			    cpy_adv[b] = advantages[idx];
			}*/
			// printf("UPDATING1.1\n");
			auto av = ac->forward(cpy_sta); // action value pairs
							// printf("UPDATING1.2\n");
			auto action = std::get<0>(av);
			auto entropy = ac->entropy().mean();
			auto new_log_prob = ac->log_prob(cpy_act);
			// printf("UPDATING1.3\n");
			auto old_log_prob = cpy_log;
			// printf("UPDATING1.3.1\n");
			// std::cout << new_log_prob.sizes() << " " << old_log_prob.sizes() << std::endl;
			auto ratio = (new_log_prob - old_log_prob).exp();
			// printf("UPDATING1.4\n");
			// std::cout << ratio.sizes() << std::endl;
			// std::cout << cpy_adv.sizes() << std::endl;
			auto surr1 = ratio * cpy_adv;
			// printf("UPDATING1.4.1\n");
			auto surr2 = torch::clamp(ratio, 1. - clip_param, 1. + clip_param) * cpy_adv;
			// printf("UPDATING1.5\n");
			auto val = std::get<1>(av);
			// printf("UPDATING1.6\n");
			auto actor_loss = -torch::min(surr1, surr2).mean();
			// printf("UPDATING1.7\n");
			auto critic_loss = torch::nn::functional::mse_loss(val, cpy_ret); //(cpy_ret - val).pow(2).mean();
			// printf("UPDATING1.8\n");
			auto loss = 0.5 * critic_loss + actor_loss - beta * entropy;
			// printf("UPDATING1.9\n");
			// std::cout << "Actor Loss: " << actor_loss.item<double>() << ", Critic Loss: " << critic_loss.item<double>() << std::endl;

			opt->zero_grad();
			// printf("UPDATING1.10\n");
			loss.backward();
			// printf("UPDATING1.11\n");
			opt->step();
			// printf("UPDATING1.12\n");
			total_loss += loss.item<double>();
			// printf("Chillin\n");
			// Sleep(10000);
			states.erase(states.begin());
			actions.erase(actions.begin());
			log_probs.erase(log_probs.begin());
			rewards.erase(rewards.begin());
			dones.erase(dones.begin());
		}
	}

	double avg_loss = total_loss / epochs;
	//std::cout << "Average training Loss: " << avg_loss << std::endl;

	c10::cuda::CUDACachingAllocator::emptyCache();

	return avg_loss;
}
