#pragma once

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
//#include <torch/nn/options/loss.h>
#include <random>
#include "NNStats.h"
#include "Models.h"
#include <Windows.h>
#include <ctime>
#include <chrono>
#include <iostream>

using uint = unsigned int;

// Vector of tensors.
using VT = std::vector<torch::Tensor>;

// Optimizer.
using OPT = torch::optim::Optimizer;

// Random engine for shuffling memory.
//std::random_device rd;
//std::mt19937 re(rd());

// Proximal policy optimization, https://arxiv.org/abs/1707.06347
class PPO
{
public:
    //static auto returns(VT& rewards, VT& dones, VT& vals, double gamma, double lambda) -> VT; // Generalized advantage estimate, https://arxiv.org/abs/1506.02438
	static auto Initilize(size_t batch_size, size_t count_players) -> void;

	static auto update(ActorCritic &ac, ActorCritic &ac_work,
		std::shared_ptr<torch::optim::Adam> &opt,
		uint steps, uint epochs, uint mini_batch_size, uint count_mini_batches,
		double ent_coef, float gamma, float lambda, c10::DeviceType device,
		NNStats &stats,
		double clip_param = .2) -> void;

    static auto save_replay(torch::Tensor &state,
	    torch::Tensor &action,
	    torch::Tensor &log_prob,
	    std::vector<float> &reward,
		std::vector<bool> &accumulation_reset,
	    std::vector<bool> &done,
		std::vector<bool> &mask,
	    bool &is_full) -> void;
    static auto count_of_replays() -> size_t;
    static auto count_of_episodes() -> size_t;
    static auto erase_player_replays(int id) -> void;
};

// Replay buffer for experience replay
class ReplayBuffer
{
public:
	ReplayBuffer(size_t capacity, size_t count_players) :
		_capacity(capacity), count_players(count_players), count_episodes(0), last_index(0), last_episode_index(0), last_values_and_returns_index(0)
	{
		//dones.resize(capacity);
		//rewards.resize(capacity);
		states = torch::empty({(long long)capacity, 40}, torch::kCUDA);
		actions = torch::empty({(long long)capacity, 5}, torch::kCUDA);
		log_probs = torch::empty({(long long)capacity, 1}, torch::kCUDA);
		values = torch::empty({(long long)capacity, 1}, torch::kCUDA);
		returns = torch::empty({(long long)capacity, 1}, torch::kCUDA);
		//v_all_indices.resize(capacity);
		players_states.resize(count_players);
		players_actions.resize(count_players);
		players_log_probs.resize(count_players);
		players_rewards.resize(count_players);
		players_accumulation_resets.resize(count_players);
		players_dones.resize(count_players);

		//local_cuda_gen = torch::make_generator<torch::CUDAGeneratorImpl>();
		//auto cuda_gen = check_generator<CUDAGeneratorImpl>(gen);
		//cuda_gen->set_current_seed(default_rng_seed_val);
		//cuda_gen->set_philox_offset_per_thread(0);
		//local_cuda_gen.set_current_seed(3112); // Local seed for randperm on GPU 3112

		//std::iota(v_all_indices.begin(), v_all_indices.end(), 0);
		//std::shuffle(v_all_indices.begin(), v_all_indices.end(), generator);
		//all_indices = torch::randperm(capacity, local_cuda_gen, torch::kCUDA);
		all_indices = torch::arange(0, capacity, torch::kCUDA);
	}

	void add(const torch::Tensor &state, const torch::Tensor &action, const torch::Tensor &log_prob, std::vector<float> &reward, std::vector<bool> &accumulation_reset, std::vector<bool> &done, std::vector<bool> &mask, bool &is_full)
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

		//int log_prob_counter = 0;
		for(size_t i = 0; i < reward.size(); i++)
		{
			if (!mask[i])
				continue;
			players_states[i].push_back(state[i]);
			players_actions[i].push_back(action[i]);
			players_log_probs[i].push_back(log_prob[i]); // log_prob_counter
			this->players_dones[i].push_back(done[i]);
			this->players_accumulation_resets[i].push_back(accumulation_reset[i]);
			this->players_rewards[i].push_back(reward[i]);
			//log_prob_counter += 1;
			if(done[i])
			{
				if(capacity() - dones.size() >= players_states[i].size())
				{
					//printf("Saving...\n");
					auto stacked = torch::stack(players_states[i]);
					//while(true)
					//{
					//	stacked = torch::stack(players_states[i]);
					//	//std::cout << stacked.sizes() << std::endl;
					//}
					//std::cout << stacked.sizes() << std::endl;
					states.index({torch::indexing::Slice(dones.size(), dones.size() + stacked.size(0))}).copy_(stacked, true);
					//states.index({(long long)(dones.size())}).copy_(stacked, true);
					players_states[i].clear();
					//printf("2\n");

					stacked = torch::stack(players_actions[i]);
					//printf("2.1\n");
					//actions.index({(long long)(dones.size())}).copy_(stacked, true);
					actions.index({torch::indexing::Slice(dones.size(), dones.size() + stacked.size(0))}).copy_(stacked, true);
					players_actions[i].clear();
					//printf("3\n");

					stacked = torch::stack(players_log_probs[i]);
					log_probs.index({torch::indexing::Slice(dones.size(), dones.size() + stacked.size(0))}).copy_(stacked, true);
					//log_probs.index({(long long)(dones.size())}).copy_(stacked, true);
					players_log_probs[i].clear();
					//printf("4\n");

					rewards.insert(rewards.end(), players_rewards[i].begin(), players_rewards[i].end());
					players_rewards[i].clear();
					accumulation_resets.insert(accumulation_resets.end(), players_accumulation_resets[i].begin(), players_accumulation_resets[i].end());
					players_accumulation_resets[i].clear();
					dones.insert(dones.end(), players_dones[i].begin(), players_dones[i].end());
					players_dones[i].clear();
					//printf("Saved.\n");
					count_episodes += 1;
				}
				else
				{
					//printf("Full\n");
					is_full = true;
				}
			}
			//printf("7\n");
		}

		/*states.index({(long long)dones.size()}).copy_(state, true);
		actions.index({(long long)dones.size()}).copy_(action, true);
		log_probs.index({(long long)dones.size()}).copy_(log_prob, true);*/
		//states.push_back(state.unsqueeze(1));
		//actions.push_back(action.unsqueeze(1));
		//log_probs.push_back(log_prob.unsqueeze(1));
		//rewards.push_back(reward);
		//std::cout << done.sizes() << std::endl;
		//dones.push_back(done);
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

		

		/*std::cout << "States size: " << states.sizes() << std::endl;
		std::cout << "Actions size: " << actions.sizes() << std::endl;
		std::cout << "Log_probs size: " << log_probs.sizes() << std::endl;
		std::cout << "Rewards size: " << rewards.sizes() << std::endl;
		std::cout << "Advantages size: " << advantages.sizes() << std::endl;*/
	}

	void erase_player_replays(int id)
	{
		this->players_states[id].clear();
		this->players_actions[id].clear();
		this->players_log_probs[id].clear();
		this->players_dones[id].clear();
		this->players_accumulation_resets[id].clear();
		this->players_rewards[id].clear();

		return;
	}

    void clear()
    {
	    // Clear the tensors by reinitializing them to empty
	    //states.clear();
	    //actions.clear();
	    //log_probs.clear();
	    //rewards.clear();
	    //dones.clear();

		//states_concatenated = torch::Tensor();
	    //actions_concat = torch::Tensor();
		//log_probs_concat = torch::Tensor();
	    //rewards_concat = torch::Tensor();
		//dones_concat = torch::Tensor();
		rewards.clear();
	    accumulation_resets.clear();
		dones.clear();
		count_episodes = last_index = last_episode_index = last_values_and_returns_index = 0;
	    //advantages.clear();

		for(size_t i = 0; i < count_players; i++)
	    {
			this->players_states[i].clear();
			this->players_actions[i].clear();
			this->players_log_probs[i].clear();
		    this->players_dones[i].clear();
			this->players_accumulation_resets[i].clear();
		    this->players_rewards[i].clear();
	    }
	    //v_all_indices.clear();
	    //v_all_indices.resize(capacity());
	    //std::iota(v_all_indices.begin(), v_all_indices.end(), 0);
	    //std::shuffle(v_all_indices.begin(), v_all_indices.end(), generator);
	    //all_indices = torch::randperm(capacity(), local_cuda_gen, torch::kCUDA);
	    all_indices = torch::arange(0, capacity(), torch::kCUDA);
    }

    size_t size()
    {
	    return dones.size();
    }

	size_t capacity()
    {
		return _capacity;
    }

	size_t episodes()
    {
	    return count_episodes;
    }

	void reset_sample_index()
    {
		last_index = 0;
	    return;
    }

	void upload_values_and_returns(torch::Tensor &values, torch::Tensor &returns)
    {
		int size = values.size(0);

		this->values.index({torch::indexing::Slice(last_values_and_returns_index, last_values_and_returns_index + size)}).copy_(values, true);
		this->returns.index({torch::indexing::Slice(last_values_and_returns_index, last_values_and_returns_index + size)}).copy_(returns, true);

		last_values_and_returns_index += size;
	    return;
    }

	bool next_episodes(
		size_t batch_size,
		torch::Tensor& states_ret,
		torch::Tensor& actions_ret,
		torch::Tensor& log_probs_ret,
		std::vector<float>& rewards_ret,
	    std::vector<bool> &accumulation_resets_ret,
		std::vector<bool>& dones_ret)
	{
		std::vector<float> rewards_concat_ret;
		std::vector<bool> accumulation_resets_concat_ret;
		std::vector<bool> dones_concat_ret;
		torch::Tensor states_concatenated_ret, actions_concat_ret, log_probs_concat_ret;

		size_t start = last_episode_index;
		size_t end = start + batch_size;
		if(start < size() && end > size())
		{
			end = size();
		}
		else if (start >= size())
		{
			return false;
		}
		// printf("1\n");
		for(size_t i = end - 1; i < size(); i++)
		{
			if(dones[i])
			{
				end = i + 1;
				break;
			}
		}
		last_episode_index = end;
		// printf("1\n");
		/*if(start != 0 && dones_concat[start - 1] != true)
		{
			for(size_t i = start; i < end; i++)
			{
				if(dones_concat[i])
				{
					start = i + 1;
					break;
				}
			}
		}*/
		// std::cout << start << " " << batch_size << std::endl;
		// printf("1\n");
		// dones_concat = dones_concat.reshape({dones_concat.numel(), 1});
		// printf("1\n");
		states_concatenated_ret = states.index({torch::indexing::Slice(start, end)});
		actions_concat_ret = actions.index({torch::indexing::Slice(start, end)});
		log_probs_concat_ret = log_probs.index({torch::indexing::Slice(start, end)});
		// rewards_concat_ret = rewards_concat.index({torch::indexing::Slice(start, end)});
		// dones_concat_ret = dones_concat.index({torch::indexing::Slice(start, end)});
		// printf("ended\n");
		// advantages_concat = advantages_concat.index({torch::indexing::Slice(start, end)});
		rewards_concat_ret = std::vector<float>(rewards.begin() + start, rewards.begin() + end);
		accumulation_resets_concat_ret = std::vector<bool>(accumulation_resets.begin() + start, accumulation_resets.begin() + end);
		dones_concat_ret = std::vector<bool>(dones.begin() + start, dones.begin() + end);
		/*std::cout << "States size: " << states_concatenated.sizes() << std::endl;
		std::cout << "Actions size: " << actions_concat.sizes() << std::endl;
		std::cout << "Log_probs size: " << log_probs_concat.sizes() << std::endl;
		std::cout << "Rewards size: " << rewards_concat.sizes() << std::endl;
		std::cout << "Advantages size: " << advantages_concat.sizes() << std::endl;
		std::cout << "Dones size: " << dones_concat.size() << std::endl;*/
		// printf("ereer\n");
		// std::cout << states.sizes() << std::endl;
		// std::sample(buffer.begin(), buffer.end(), std::back_inserter(batch), batch_size, std::mt19937{std::random_device{}()});
		states_ret = states_concatenated_ret;
		actions_ret = actions_concat_ret;
		log_probs_ret = log_probs_concat_ret;
		rewards_ret = rewards_concat_ret;
		accumulation_resets_ret = accumulation_resets_concat_ret;
		dones_ret = dones_concat_ret;
		return true;
	}

	void clear_last_sample_index_counter()
	{
		last_index = 0;
	}

	// MAE - Mean Absolute Error
	// float mae = 0.0f;
	// for(size_t i = 0; i < critic_predictions.size(); ++i)
	// {
	//     mae += std::abs(critic_predictions[i] - actual_returns[i]);
	// }
	// mae /= critic_predictions.size();
	//
	//
	double get_mae()
	{
		double mae = 0.0f;

		// Compute MAE directly without intermediate tensors
		mae = torch::nn::functional::l1_loss(values, returns, torch::nn::L1LossOptions().reduction(torch::kMean))
			      .item<double>();

		return mae;
	}

	// Correlation coefficient
	// Between values and actual returns
	float get_correlation_coefficient()
	{
		// Ensure inputs are 1D and have the same size
		if(values.dim() != returns.dim() || returns.size(0) != values.size(0))
		{
			throw std::invalid_argument("Inputs must be with same dimensions of tensors of the same size.");
		}

		// Compute means
		float mean_values = values.mean().item<float>();
		float mean_returns = returns.mean().item<float>();

		// Compute covariance
		auto cov = ((values - mean_values) * (returns - mean_returns)).mean().item<float>();

		// Compute standard deviations
		float std_values = values.std().item<float>();
		float std_returns = returns.std().item<float>();

		// Compute correlation coefficient
		float r = cov / (std_values * std_returns);
		return r;
	}

	bool next_sample(
		size_t batch_size,
		torch::Tensor &states_ret,
		torch::Tensor &actions_ret,
		torch::Tensor &log_probs_ret,
		torch::Tensor &values_ret,
		torch::Tensor &returns_ret
	)
	{
		//std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<bool>, torch::Tensor>> batch;
		//printf("1\n");

		//if(dones_concat.size() == 0)
		//{
		//	// printf("creating\n");
		//	// Sleep(3000);
		//	//states_concatenated = torch::cat(states, 1);
		//	// printf("states catted\n");
		//	// Sleep(7000);
		//	// states_concatenated = torch::Tensor();
		//	// printf("states deleted\n");
		//	// Sleep(7000);
		//	//printf("1\n");
		//	//actions_concat = torch::cat(actions, 1);
		//	//printf("1\n");
		//	//log_probs_concat = torch::cat(log_probs, 1).detach();
		//	//printf("1\n");
		//	//rewards_concat = torch::cat(rewards, 1);
		//	//std::cout << rewards[0].sizes() << std::endl;
		//	//std::cout << dones[0].sizes() << std::endl;
		//	//dones_concat = torch::cat(dones, 1);
		//	// printf("created\n");
		//	//  advantages_concat = torch::cat(rewards, 1);
		//	//printf("1\n");

		//	states_concatenated_reshaped = states_concatenated.view({states_concatenated.sizes()[0] * states_concatenated.sizes()[1], states_concatenated.sizes()[2]});
		//	actions_concat_reshaped = actions_concat.view({actions_concat.sizes()[0] * actions_concat.sizes()[1], actions_concat.sizes()[2]});
		//	//printf("1\n");
		//	log_probs_concat_reshaped = log_probs_concat.view({log_probs_concat.sizes()[0] * log_probs_concat.sizes()[1], log_probs_concat.sizes()[2]});
		//	//printf("1\n");
		//	// std::cout << "Rewards size: " << rewards.sizes() << " " << rewards.size(0) << std::endl;
		//	//rewards_concat = rewards_concat.reshape({rewards_concat.numel(), 1});
		//	//dones_concat = dones_concat.reshape({dones_concat.numel(), 1});
		//	//printf("1\n");
		//	// advantages_concat = advantages_concat.reshape({advantages_concat.numel()});
		//	// printf("1\n");
		//	for(size_t i = 0; i < count_players; i++)
		//	{
		//		rewards_concat.insert(rewards_concat.end(), rewards[i].begin(), rewards[i].end());
		//		dones_concat.insert(dones_concat.end(), dones[i].begin(), dones[i].end());
		//	}
		//}

		//printf("1\n");
		torch::Tensor states_concatenated_ret, actions_concat_ret, log_probs_concat_ret, values_concat_ret, returns_concat_ret;

		size_t start = last_index;
		size_t end = start + batch_size;
		if(end > size())
		{
			return false;
		}
		for(size_t i = end - 1; i < size(); i++)
		{
			if(dones[i])
			{
				end = i + 1;
				break;
			}
		}
		last_index = end;
		//printf("1\n");
		/*if(start != 0 && dones_concat[start - 1] != true)
		{
			for(size_t i = start; i < end; i++)
			{
				if(dones_concat[i])
				{
					start = i + 1;
					break;
				}
			}
		}*/
		//std::cout << start << " " << batch_size << std::endl;
		//printf("1\n");
		//dones_concat = dones_concat.reshape({dones_concat.numel(), 1});
		//printf("1\n");

		/*states_concatenated_ret = states.index({torch::indexing::Slice(start, end)});
		actions_concat_ret = actions.index({torch::indexing::Slice(start, end)});
		log_probs_concat_ret = log_probs.index({torch::indexing::Slice(start, end)});
		values_concat_ret = values.index({torch::indexing::Slice(start, end)});
		returns_concat_ret = returns.index({torch::indexing::Slice(start, end)});*/

		auto group_indices = all_indices.slice(0, start, end);

		states_concatenated_ret = states.index_select(0, group_indices);
		actions_concat_ret = actions.index_select(0, group_indices);
		log_probs_concat_ret = log_probs.index_select(0, group_indices);
		values_concat_ret = values.index_select(0, group_indices);
		returns_concat_ret = returns.index_select(0, group_indices);

		//rewards_concat_ret = rewards_concat.index({torch::indexing::Slice(start, end)});
		//dones_concat_ret = dones_concat.index({torch::indexing::Slice(start, end)});
		//printf("ended\n");
		//advantages_concat = advantages_concat.index({torch::indexing::Slice(start, end)});
		//rewards_concat_ret = std::vector<float>(rewards.begin() + start, rewards.begin() + end);
		//dones_concat_ret = std::vector<bool>(dones.begin() + start, dones.begin() + end);
		/*std::cout << "States size: " << states_concatenated.sizes() << std::endl;
		std::cout << "Actions size: " << actions_concat.sizes() << std::endl;
		std::cout << "Log_probs size: " << log_probs_concat.sizes() << std::endl;
		std::cout << "Rewards size: " << rewards_concat.sizes() << std::endl;
		std::cout << "Advantages size: " << advantages_concat.sizes() << std::endl;
		std::cout << "Dones size: " << dones_concat.size() << std::endl;*/
		//printf("ereer\n");
		//std::cout << states.sizes() << std::endl;
		//std::sample(buffer.begin(), buffer.end(), std::back_inserter(batch), batch_size, std::mt19937{std::random_device{}()});
		states_ret = states_concatenated_ret;
		actions_ret = actions_concat_ret;
		log_probs_ret = log_probs_concat_ret;
		values_ret = values_concat_ret;
		returns_ret = returns_concat_ret;
		return true;
	}

private:
	size_t last_index, last_episode_index, last_values_and_returns_index;
	size_t count_episodes;
	size_t count_players;
	size_t _capacity;
	torch::Tensor states, actions, log_probs, values, returns, all_indices;
	//torch::Tensor states_concatenated_reshaped, actions_concat_reshaped, log_probs_concat_reshaped;
	//std::vector<int64_t> v_all_indices;
	std::vector<float> rewards;
	std::vector<bool> accumulation_resets;
	std::vector<bool> dones;
	std::vector<std::vector<float>> players_rewards;
	std::vector<std::vector<bool>> players_dones, players_accumulation_resets;
	std::vector<std::vector<torch::Tensor>> players_states, players_actions, players_log_probs;

	//std::mt19937 generator{std::random_device{}()};
	//torch::Generator local_cuda_gen;
	//std::vector<torch::Tensor> /*states,*/ actions, log_probs;
	//std::vector<std::vector<float>> rewards;
	//std::vector<std::vector<bool>> dones;
	//std::tuple < torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<std::vector<bool>>, torch::Tensor> buffer;
	//torch::Tensor actions, log_probs, rewards, advantages;
	//std::vector<std::vector<bool>> dones;
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

torch::Tensor normalize_advantages(const torch::Tensor &advantages)
{
	// Detach to prevent gradients flowing through normalization
	auto advantages_detached = advantages.detach();

	// Compute mean and standard deviation
	auto mean = advantages_detached.mean();
	auto std = advantages_detached.std();

	// Normalize with epsilon for numerical stability
	return (advantages - mean) / (std + 1e-8);
}

torch::Tensor compute_advantages(ActorCritic &ac, const torch::Tensor &returns, const torch::Tensor &values)
{
	//auto values = ac->critic_forward(states);
	//std::cout << values.sizes() << std::endl;
	torch::Tensor advantages = returns - values;
	return advantages; // normalize_advantages(advantages);
}

std::vector<float> tensor_to_vector(const torch::Tensor &tensor)
{
	auto tensor_cpu = tensor.to(torch::kCPU);

	// Ensure the tensor is of type float and is contiguous
	torch::Tensor contiguous_tensor = tensor_cpu.contiguous();

	// Get the number of elements in the tensor
	int64_t num_elements = contiguous_tensor.numel();

	// Get the raw data pointer from the tensor
	float *data_ptr = contiguous_tensor.data_ptr<float>();

	// Create a vector from the raw data pointer
	std::vector<float> vec(data_ptr, data_ptr + num_elements);

	return vec;
}

// Function to compute the mean of the vector
float mean(const std::vector<float> &vec)
{
	return std::accumulate(vec.begin(), vec.end(), 0.0f) / vec.size();
}

// Function to compute the standard deviation of the vector
float standard_deviation(const std::vector<float> &vec, float mean_val)
{
	float sum = 0.0f;
	for(float val : vec)
	{
		sum += (val - mean_val) * (val - mean_val);
	}
	return std::sqrt(sum / vec.size());
}

// Function to standardize rewards
std::vector<float> standardize_rewards(const std::vector<float> &rewards)
{
	float mean_val = mean(rewards);
	float stddev_val = standard_deviation(rewards, mean_val);

	std::vector<float> standardized_rewards;
	for(float reward : rewards)
	{
		float standardized_reward = (reward - mean_val) / (stddev_val + 1e-8); // Adding small epsilon to avoid division by zero
		standardized_rewards.push_back(standardized_reward);
	}

	return standardized_rewards;
}

torch::Tensor calculate_returns(std::vector<float> &rewards, std::vector<bool> &accumulation_resets, std::vector<bool> &dones, const torch::Tensor &values, float gamma, float lambda)
{
	//printf("FINNNN1.1\n");

	float gae = 0;
	std::vector<float> returns(rewards.size());
	std::vector<float> vValues = tensor_to_vector(values);

	//rewards = standardize_rewards(rewards);

	for(int64_t i = rewards.size() - 1; i >= 0; --i)
	{
		float delta = 0;
		if(i == rewards.size() - 1)
			delta = rewards[i] + gamma * vValues[i] * (1 - (dones[i] || accumulation_resets[i])) - vValues[i];
		else
			delta = rewards[i] + gamma * vValues[i + 1] * (1 - (dones[i] || accumulation_resets[i])) - vValues[i];

		gae = delta + gamma * lambda * (1 - (dones[i] || accumulation_resets[i])) * gae;
		// printf("FINNNN1.4\n");
		// G = rewards[i] + gamma * G;
		// std::cout << G.item<float>() << std::endl;
		// printf("FINNNN1.5\n");
		returns[i] = gae + vValues[i];
		// printf("FINNNN1.6\n");
	}

	//torch::Tensor returns = torch::zeros({rewards.size(0), 1}, torch::kCPU);
	////printf("FINNNN1.2\n");
	//torch::Tensor G = torch::zeros({1}, torch::kCPU);
	////printf("FINNNN1.3\n");
	////std::cout << returns.sizes() << std::endl;
	////std::cout << G.sizes() << std::endl;

	//for(int64_t i = rewards.size(0) - 1; i >= 0; --i)
	//{
	//	G = rewards[i] + gamma * G * (!dones[i]);
	//	//printf("FINNNN1.4\n");
	//	//G = rewards[i] + gamma * G;
	//	//std::cout << G.item<float>() << std::endl;
	//	//printf("FINNNN1.5\n");
	//	returns[i] = G;
	//	//printf("FINNNN1.6\n");
	//}

	torch::Tensor tRet = torch::from_blob(returns.data(), {(long long)returns.size(), 1}, torch::kF32);
	//auto decide_time = std::chrono::high_resolution_clock::now();
	
	tRet = tRet.to(torch::kCUDA, true);
	/*auto now = std::chrono::high_resolution_clock::now();
	std::cout << "Time to transfer: " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(now - decide_time).count()) << std::endl;*/
	//tDones = tDones.to(device, myStream);

	//printf("FINE\n");
	//std::cout << rewards.sizes() << std::endl;
	//std::cout << returns.sizes() << std::endl;

	return tRet.detach();
}

auto PPO::save_replay(torch::Tensor& state,
    torch::Tensor& action,
    torch::Tensor& log_prob,
    std::vector<float> &reward,
	std::vector<bool> &accumulation_reset,
	std::vector<bool> &done,
	std::vector<bool> &mask,
	bool &is_full) -> void
{
	//torch::NoGradGuard no_grad;
	replay_buffer->add(state, action, log_prob, reward, accumulation_reset, done, mask, is_full);
}

auto PPO::erase_player_replays(int id) -> void
{
	replay_buffer->erase_player_replays(id);
	return;
}

auto PPO::count_of_replays() -> size_t
{
	return replay_buffer->size();
}

auto PPO::count_of_episodes() -> size_t
{
	return replay_buffer->episodes();
}

auto PPO::update(ActorCritic &ac, ActorCritic &ac_work,
	std::shared_ptr<torch::optim::Adam> &opt,
	uint steps, uint epochs, uint mini_batch_size, uint count_mini_batches,
	double ent_coef, float gamma, float lambda, c10::DeviceType device,
	NNStats& stats,
	double clip_param) -> void
{
	torch::Tensor total_loss_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate loss
	torch::Tensor total_actor_loss_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate actor loss
	torch::Tensor total_critic_loss_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate critic loss
	torch::Tensor total_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate entropy
	//torch::Tensor max_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to save max entropy
	//torch::Tensor median_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to save median entropy
	//torch::Tensor mode_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to save mode entropy

	torch::Tensor total_angle_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate angle entropy
	torch::Tensor total_hook_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate hook entropy
	torch::Tensor total_hammer_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate hammer entropy
	torch::Tensor total_direction_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate direction entropy

	torch::Tensor total_actor_grad_norm = torch::zeros({}, torch::kCUDA);
	torch::Tensor total_critic_grad_norm = torch::zeros({}, torch::kCUDA);
	torch::Tensor total_actor_weight_norm = torch::zeros({}, torch::kCUDA);
	torch::Tensor total_critic_weight_norm = torch::zeros({}, torch::kCUDA);
	torch::Tensor total_actor_activation_mean = torch::zeros({}, torch::kCUDA);
	torch::Tensor total_actor_activation_std = torch::zeros({}, torch::kCUDA);

	torch::Tensor min_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate entropy
	torch::Tensor min_angle_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate angle entropy
	torch::Tensor min_hook_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate hook entropy
	torch::Tensor min_hammer_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate hammer entropy
	torch::Tensor min_direction_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate direction entropy

	torch::Tensor max_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate entropy
	torch::Tensor max_angle_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate angle entropy
	torch::Tensor max_hook_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate hook entropy
	torch::Tensor max_hammer_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate hammer entropy
	torch::Tensor max_direction_entropy_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate direction entropy

	auto saved_size = replay_buffer->size();
	size_t count_updates = 0;
	int count_mini_batches_processed = 0;

	{
		// Wait for all log probs to come to cpu
		at::cuda::getCurrentCUDAStream().synchronize();
		//printf("2\n");
		//Sleep(5000);
		
		//printf("CHECK\n");
		// Sleep(10000);
		// printf("CLEARING\n");
		// torch::cuda::synchronize();
		// Sleep(10000);
		//c10::cuda::CUDACachingAllocator::emptyCache();

		//opt->zero_grad();
		torch::Tensor states, actions, log_probs;
		std::vector<float> rewards;
		std::vector<bool> accumulation_resets;
		std::vector<bool> dones;
		//printf("Calculating GAE of episodes...");
		while(replay_buffer->next_episodes(mini_batch_size, states, actions, log_probs, rewards, accumulation_resets, dones))
		{
			torch::Tensor cpy_values = ac_work->critic_forward(states).detach();

			auto returns = calculate_returns(rewards, accumulation_resets, dones, cpy_values, gamma, lambda);

			replay_buffer->upload_values_and_returns(cpy_values, returns);
		}
		//printf(" done!\n");
		stats.critic_mean_absolute_error = replay_buffer->get_mae();
		stats.critic_correlation_coefficient = replay_buffer->get_correlation_coefficient();
		
		for(size_t i = 0; i < epochs; i++)
		{
			replay_buffer->clear_last_sample_index_counter();

			torch::Tensor values;
			torch::Tensor returns;
			while(replay_buffer->next_sample(mini_batch_size, states, actions, log_probs, values, returns))
			{
				//torch::Tensor states_cpy = states;
				//torch::Tensor log_probs_cpy = log_probs.detach();

				torch::Tensor cpy_sta = states;
			
				//cpy_sta = states_cpy;
				// torch::Tensor cpy_values = ac_work->critic_forward(cpy_sta).detach();
				torch::Tensor cpy_values = values;

				// std::cout << cpy_sta.sizes() << std::endl;
				// printf("UPDATING0.1.1\n");
				torch::Tensor cpy_act = actions;
				// printf("UPDATING0.1.2\n");
				torch::Tensor cpy_log = log_probs.detach();

				torch::Tensor cpy_ret = returns; // normalize_rewards(returnsee);
				
				torch::Tensor cpy_adv = compute_advantages(ac, cpy_ret, cpy_values /*cpy_sta.view({cpy_sta.size(0), 1, 4372})*/);

				// printf("UPDATING1.1\n");
				torch::Tensor action = ac->actor_forward(cpy_sta);
				//printf("4\n");
				//Sleep(3000);
				//std::cout << action.slice(0, 0, 10) << std::endl;
				//torch::Tensor entropy = ac->entropy(action).mean();
				//std::cout << action.slice(0, 0, 10) << std::endl;

				// Bound it between lower_bound and upper_bound:
				double lower_bound = -4.0;
				double upper_bound = 0.0;
				auto log_std = lower_bound + (upper_bound - lower_bound) * ((torch::tanh(action.slice(1, 7, 9)) + 1) / 2);
				//auto log_std = action.slice(1, 7, 9);
				//std::cout << log_std.sizes() << std::endl;
				//auto log_std_penalty = torch::relu(action.slice(1, 7, 9) - 2);
				//auto log_std = action.slice(1, 7, 9)/*.clamp_max(0)*/;

				auto angle_entropy = ac->entropy_gaussian(log_std) / (1.42 * 2); // 1.42 * 2
				//std::cout << angle_entropy.sizes() << std::endl;


				auto probs = action.slice(1, 5, 6); // Shape [batch_size, 1]
				auto hook_entropy = ac->entropy_bernoulli(probs) / log(2); // log(2)
				probs = torch::sigmoid(action.slice(1, 6, 7)); // Shape [batch_size, 1]
				auto hammer_entropy = ac->entropy_bernoulli(probs) / log(2); // log(2)

				probs = torch::softmax(action.slice(1, 2, 5), 1); // Shape [batch_size, 3]
				auto direction_entropy = ac->entropy_categorical(probs) / log(3); // log(3)

				hook_entropy = hook_entropy.squeeze(-1); // Convert from [batch_size, 1] to [batch_size]
				hammer_entropy = hammer_entropy.squeeze(-1);

				//max_entropy_tensor = angle_entropy.max();
				//median_entropy_tensor = angle_entropy.median();
				//mode_entropy_tensor = angle_entropy.mode(0);

				torch::Tensor _entropy = angle_entropy * 0.625 + hook_entropy + hammer_entropy + direction_entropy;
				torch::Tensor entropy = _entropy.mean();

				//printf("calculated\n");

				//Sleep(6000);

				//std::cout << entropy << std::endl;
				// printf("UPDATING1.3\n");
				torch::Tensor new_log_prob = ac->log_prob(action, cpy_act);
				//printf("calculated\n");
				//Sleep(6000);
				// printf("UPDATING1.4\n");
				torch::Tensor old_log_prob = cpy_log;
				//std::cout << "Begin" << std::endl;
				//std::cout << new_log_prob.slice(0, 0, 10) << std::endl;
				//std::cout << old_log_prob.slice(0, 0, 10) << std::endl;
				// printf("UPDATING1.4.1\n");
				//std::cout << new_log_prob.sizes() << " " << old_log_prob.sizes() << std::endl;
				auto ratio = (new_log_prob - old_log_prob).exp();
				//std::cout << "Ratio mean: " << ratio.mean().item<double>() << ", std: " << ratio.std().item<double>() << std::endl;
				//std::cout << "Ratio min: " << ratio.min().item<double>() << ", max: " << ratio.max().item<double>() << std::endl;
				// printf("UPDATING1.5\n");
				//std::cout << ratio.sizes() << std::endl;
				//std::cout << cpy_adv.sizes() << std::endl;
				auto surr1 = ratio * cpy_adv;
				// printf("UPDATING1.5.1\n");
				auto surr2 = torch::clamp(ratio, 1. - clip_param, 1. + clip_param) * cpy_adv;
				// printf("UPDATING1.6\n");
				// printf("4.9\n");
				// Sleep(7000);
				auto val = ac->critic_forward(cpy_sta);
				// printf("5\n");
				// Sleep(7000);
				auto actor_loss = -torch::min(surr1, surr2).mean();
				// printf("UPDATING1.7\n");
				auto critic_loss = torch::nn::functional::mse_loss(val, cpy_ret); //(cpy_ret - val).pow(2).mean();
				// printf("UPDATING1.8\n");
				auto loss = 0.5 * critic_loss + actor_loss - ent_coef * entropy;
				//loss /= count_mini_batches;

				// printf("UPDATING1.9\n");
				//  std::cout << "Actor Loss: " << actor_loss.item<double>() << ", Critic Loss: " << critic_loss.item<double>() << std::endl;

				// printf("UPDATING1.10\n");
				try
				{
					loss.backward();
				}
				catch(const std::exception &e)
				{
					std::cout << "Exception during backward pass: " << e.what() << std::endl;
				}

				// Global gradient clipping specific to PPO
				torch::nn::utils::clip_grad_norm_(ac->parameters(), 0.5f);
				//torch::nn::utils::clip_grad_norm_(ac->critic_network->parameters(), 0.5f);

				// Compute gradient norms
				double actor_grad_norm = 0.0;
				double critic_grad_norm = 0.0;
				for(const auto &param : ac->actor_parameters())
				{
					if(param.grad().defined())
					{
						actor_grad_norm += param.grad().norm().item<double>();
					}
				}
				for(const auto &param : ac->critic_parameters())
				{
					if(param.grad().defined())
					{
						critic_grad_norm += param.grad().norm().item<double>();
					}
				}
				total_actor_grad_norm += actor_grad_norm;
				total_critic_grad_norm += critic_grad_norm;

				// Compute weight norms
				double actor_weight_norm = 0.0;
				double critic_weight_norm = 0.0;
				for(const auto &param : ac->actor_parameters())
				{
					actor_weight_norm += param.norm().item<double>();
				}
				for(const auto &param : ac->critic_parameters())
				{
					critic_weight_norm += param.norm().item<double>();
				}
				total_actor_weight_norm += actor_weight_norm;
				total_critic_weight_norm += critic_weight_norm;

				// Compute activation statistics
				auto actor_activations = ac->actor_forward(cpy_sta);
				auto actor_activation_mean = actor_activations.mean().item<double>();
				auto actor_activation_std = actor_activations.std().item<double>();
				total_actor_activation_mean += actor_activation_mean;
				total_actor_activation_std += actor_activation_std;

				count_mini_batches_processed += 1;
				if(count_mini_batches_processed == count_mini_batches)
				{
					opt->step();
					opt->zero_grad();

					// Clamp log_std_ so it will not rise infinetly
					//{
					//	//torch::NoGradGuard no_grad; // Disable gradient tracking
					//	ac->log_std_.clamp_(-3.0, 0); // -3.0 0
					//}
					count_mini_batches_processed = 0;
				}
				if(ac->actor_network->parameters()[0].isnan().any().item<bool>() || ac->actor_network->parameters()[0].isinf().any().item<bool>())
				{
					std::cout << "Nan or inf detected in actor_network parameters" << std::endl;
					std::cout << states.sizes() << std::endl;
					std::cout << actions.sizes() << std::endl;
					std::cout << log_probs.sizes() << std::endl;
					std::cout << values.sizes() << std::endl;
					std::cout << returns.sizes() << std::endl;
					std::cout << entropy << std::endl;
					std::cout << actor_loss << std::endl;
					std::cout << critic_loss << std::endl;
					std::cout << loss << std::endl;
					std::cout << "States - Mean: " << states.mean().item<double>()
						  << ", Std: " << states.std().item<double>()
						  << ", Min: " << states.min().item<double>()
						  << ", Max: " << states.max().item<double>()
						  << ", NaN Count: " << torch::isnan(states).sum().item<int64_t>()
						  << ", Inf Count: " << torch::isinf(states).sum().item<int64_t>() << std::endl;
					std::cout << "States (First 5 elements): " << states.index({torch::indexing::Slice(0, 5)}) << std::endl;

					std::cout << "Actions - Mean: " << actions.mean().item<double>()
						  << ", Std: " << actions.std().item<double>()
						  << ", Min: " << actions.min().item<double>()
						  << ", Max: " << actions.max().item<double>()
						  << ", NaN Count: " << torch::isnan(actions).sum().item<int64_t>()
						  << ", Inf Count: " << torch::isinf(actions).sum().item<int64_t>() << std::endl;
					std::cout << "Actions (First 5 elements): " << actions.index({torch::indexing::Slice(0, 2000)}) << std::endl;

					// Find the maximum value and the index of the maximum value
					torch::Tensor max_value_tensor = std::get<0>(actions.max(1)); // Max value along the first dimension (rows)
					int64_t max_index = max_value_tensor.argmax().item<int64_t>(); // Index of max value

					// Print the row with the maximum value
					std::cout << "Row index with max value: " << max_index << std::endl;
					std::cout << "Values in that row: " << actions[max_index] << std::endl;

					std::cout << "Values - Mean: " << values.mean().item<double>()
						  << ", Std: " << values.std().item<double>()
						  << ", Min: " << values.min().item<double>()
						  << ", Max: " << values.max().item<double>()
						  << ", NaN Count: " << torch::isnan(values).sum().item<int64_t>()
						  << ", Inf Count: " << torch::isinf(values).sum().item<int64_t>() << std::endl;

					std::cout << "Returns - Mean: " << returns.mean().item<double>()
						  << ", Std: " << returns.std().item<double>()
						  << ", Min: " << returns.min().item<double>()
						  << ", Max: " << returns.max().item<double>()
						  << ", NaN Count: " << torch::isnan(returns).sum().item<int64_t>()
						  << ", Inf Count: " << torch::isinf(returns).sum().item<int64_t>() << std::endl;

					std::cout << "Old Log Prob - Mean: " << old_log_prob.mean().item<double>()
						  << ", Std: " << old_log_prob.std().item<double>()
						  << ", Min: " << old_log_prob.min().item<double>()
						  << ", Max: " << old_log_prob.max().item<double>() << std::endl;

					std::cout << "New Log Prob - Mean: " << new_log_prob.mean().item<double>()
						  << ", Std: " << new_log_prob.std().item<double>()
						  << ", Min: " << new_log_prob.min().item<double>()
						  << ", Max: " << new_log_prob.max().item<double>() << std::endl;

					std::cout << "Advantages - Mean: " << cpy_adv.mean().item<double>()
						  << ", Std: " << cpy_adv.std().item<double>()
						  << ", Min: " << cpy_adv.min().item<double>()
						  << ", Max: " << cpy_adv.max().item<double>() << std::endl;

					for(const auto &param : ac->actor_parameters())
					{
						if(param.grad().defined())
						{
							std::cout << "Actor Grad - Mean: " << param.grad().mean().item<double>()
								  << ", Std: " << param.grad().std().item<double>()
								  << ", Min: " << param.grad().min().item<double>()
								  << ", Max: " << param.grad().max().item<double>() << std::endl;
						}
					}

					/*std::cout << "Entropy: " << entropy.item<double>()
						  << ", Log Std - Mean: " << ac->log_std_.mean().item<double>()
						  << ", Std: " << ac->log_std_.std().item<double>()
						  << ", Min: " << ac->log_std_.min().item<double>()
						  << ", Max: " << ac->log_std_.max().item<double>() << std::endl;*/

					auto clipped_ratio = torch::clamp(ratio, 1. - clip_param, 1. + clip_param);
					std::cout << "Clipped Ratio - Mean: " << clipped_ratio.mean().item<double>()
						  << ", Std: " << clipped_ratio.std().item<double>()
						  << ", Min: " << clipped_ratio.min().item<double>()
						  << ", Max: " << clipped_ratio.max().item<double>() << std::endl;
					system("PAUSE");

				}
				// torch::nn::utils::clip_grad_norm_(ac->parameters(), 1.0); // Clip gradients
				// printf("UPDATING1.11\n");
				// bb = ac->normal_actor(action);

				// printf("UPDATING1.12\n");
				total_actor_loss_tensor += actor_loss.detach();
				total_critic_loss_tensor += critic_loss.detach();
				total_loss_tensor += loss.detach();
				total_entropy_tensor += entropy.detach();

				total_angle_entropy_tensor += angle_entropy.detach().mean();
				total_hook_entropy_tensor += hook_entropy.detach().mean();
				total_hammer_entropy_tensor += hammer_entropy.detach().mean();
				total_direction_entropy_tensor += direction_entropy.detach().mean();

				min_entropy_tensor += _entropy.detach().min();
				min_angle_entropy_tensor += angle_entropy.detach().min();
				min_hook_entropy_tensor += hook_entropy.detach().min();
				min_hammer_entropy_tensor += hammer_entropy.detach().min();
				min_direction_entropy_tensor += direction_entropy.detach().min();

				max_entropy_tensor += _entropy.detach().max();
				max_angle_entropy_tensor += angle_entropy.detach().max();
				max_hook_entropy_tensor += hook_entropy.detach().max();
				max_hammer_entropy_tensor += hammer_entropy.detach().max();
				max_direction_entropy_tensor += direction_entropy.detach().max();

				count_updates += 1;
				//c10::cuda::CUDACachingAllocator::emptyCache();

				// printf("Pre next\n");
				// Sleep(5000);
			}
		}
	}
	double avg_loss = 0;
	//auto decide_time = std::chrono::high_resolution_clock::now();
	replay_buffer->clear();

	stats.avg_training_loss = total_loss_tensor.item<double>() / count_updates;
	stats.avg_actor_loss = total_actor_loss_tensor.item<double>() / count_updates;
	stats.avg_critic_loss = total_critic_loss_tensor.item<double>() / count_updates;
	stats.avg_entropy = total_entropy_tensor.item<double>() / count_updates;

	stats.avg_actor_grad_norm = total_actor_grad_norm.item<double>() / count_updates;
	stats.avg_critic_grad_norm = total_critic_grad_norm.item<double>() / count_updates;
	stats.avg_actor_weight_norm = total_actor_weight_norm.item<double>() / count_updates;
	stats.avg_critic_weight_norm = total_critic_weight_norm.item<double>() / count_updates;
	stats.avg_actor_activation_mean = total_actor_activation_mean.item<double>() / count_updates;
	stats.avg_actor_activation_std = total_actor_activation_std.item<double>() / count_updates;

	stats.avg_angle_entropy = total_angle_entropy_tensor.item<double>() / count_updates;
	stats.avg_hook_entropy = total_hook_entropy_tensor.item<double>() / count_updates;
	stats.avg_hammer_entropy = total_hammer_entropy_tensor.item<double>() / count_updates;
	stats.avg_direction_entropy = total_direction_entropy_tensor.item<double>() / count_updates;

	stats.min_entropy = min_entropy_tensor.item<double>() / count_updates;
	stats.min_angle_entropy = min_angle_entropy_tensor.item<double>() / count_updates;
	stats.min_hook_entropy = min_hook_entropy_tensor.item<double>() / count_updates;
	stats.min_hammer_entropy = min_hammer_entropy_tensor.item<double>() / count_updates;
	stats.min_direction_entropy = min_direction_entropy_tensor.item<double>() / count_updates;

	stats.max_entropy = max_entropy_tensor.item<double>() / count_updates;
	stats.max_angle_entropy = max_angle_entropy_tensor.item<double>() / count_updates;
	stats.max_hook_entropy = max_hook_entropy_tensor.item<double>() / count_updates;
	stats.max_hammer_entropy = max_hammer_entropy_tensor.item<double>() / count_updates;
	stats.max_direction_entropy = max_direction_entropy_tensor.item<double>() / count_updates;

	//std::cout << "Max entropy: " << max_entropy_tensor << std::endl;
	//std::cout << "Median entropy: " << median_entropy_tensor << std::endl;

	//auto now = std::chrono::high_resolution_clock::now();
	//std::cout << "Time to calculate loss: " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(now - decide_time).count()) << std::endl;
	//std::cout << "Average training Loss: " << avg_loss << std::endl;

	//c10::cuda::CUDACachingAllocator::emptyCache();

	return;
}
