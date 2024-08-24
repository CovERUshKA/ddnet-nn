#pragma once

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <random>

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
std::random_device rd;
std::mt19937 re(rd());

// Proximal policy optimization, https://arxiv.org/abs/1707.06347
class PPO
{
public:
    //static auto returns(VT& rewards, VT& dones, VT& vals, double gamma, double lambda) -> VT; // Generalized advantage estimate, https://arxiv.org/abs/1506.02438
	static auto Initilize(size_t batch_size, size_t count_players) -> void;
    static auto update(ActorCritic& ac,
	    std::shared_ptr<torch::optim::Adam> &opt, 
                       uint steps, uint epochs, uint mini_batch_size, double beta, float gamma, float lambda, c10::DeviceType device, double &avg_training_loss, double &avg_actor_loss, double &avg_critic_loss, double clip_param = .2) -> void;
    static auto save_replay(torch::Tensor &state,
	    torch::Tensor &action,
	    torch::Tensor &log_prob,
	    std::vector<float> &reward,
	    std::vector<bool> &done) -> void;
    static auto count_of_replays() -> size_t;
};

// Replay buffer for experience replay
class ReplayBuffer
{
public:
	ReplayBuffer(size_t capacity, size_t count_players) :
		_capacity(capacity), count_players(count_players)
	{
		dones.resize(count_players);
		rewards.resize(count_players);
		states_concatenated = torch::empty({(long long)count_players, (long long)(capacity / count_players), 1167}, torch::kCUDA);
		actions_concat = torch::empty({(long long)count_players, (long long)(capacity / count_players), 9}, torch::kCUDA);
		log_probs_concat = torch::empty({(long long)count_players, (long long)(capacity / count_players), 9}, torch::kCUDA);
	}

	void add(const torch::Tensor &state, const torch::Tensor &action, const torch::Tensor &log_prob, std::vector<float> &reward, std::vector<bool> &done)
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

		states_concatenated.index({torch::indexing::Slice(), (long long)dones[0].size()}).copy_(state, true);
		actions_concat.index({torch::indexing::Slice(), (long long)dones[0].size()}).copy_(action, true);
		log_probs_concat.index({torch::indexing::Slice(), (long long)dones[0].size()}).copy_(log_prob, true);
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

		for(size_t i = 0; i < count_players; i++)
		{
			this->dones[i].push_back(done[i]);
			this->rewards[i].push_back(reward[i]);
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
		rewards_concat.clear();
		dones_concat.clear();
	    //advantages.clear();

		for(size_t i = 0; i < count_players; i++)
	    {
		    this->dones[i].clear();
		    this->rewards[i].clear();
	    }
    }

    size_t size()
    {
	    return dones[0].size() * count_players;
    }

	size_t capacity()
    {
		return _capacity;
    }

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<float>, std::vector<bool>> sample(size_t batch_size, int i)
	{
		//std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<bool>, torch::Tensor>> batch;
		//printf("1\n");

		if(dones_concat.size() == 0)
		{
			// printf("creating\n");
			// Sleep(3000);
			//states_concatenated = torch::cat(states, 1);
			// printf("states catted\n");
			// Sleep(7000);
			// states_concatenated = torch::Tensor();
			// printf("states deleted\n");
			// Sleep(7000);
			//printf("1\n");
			//actions_concat = torch::cat(actions, 1);
			//printf("1\n");
			//log_probs_concat = torch::cat(log_probs, 1).detach();
			//printf("1\n");
			//rewards_concat = torch::cat(rewards, 1);
			//std::cout << rewards[0].sizes() << std::endl;
			//std::cout << dones[0].sizes() << std::endl;
			//dones_concat = torch::cat(dones, 1);
			// printf("created\n");
			//  advantages_concat = torch::cat(rewards, 1);
			//printf("1\n");

			states_concatenated_reshaped = states_concatenated.view({states_concatenated.sizes()[0] * states_concatenated.sizes()[1], states_concatenated.sizes()[2]});
			actions_concat_reshaped = actions_concat.view({actions_concat.sizes()[0] * actions_concat.sizes()[1], actions_concat.sizes()[2]});
			//printf("1\n");
			log_probs_concat_reshaped = log_probs_concat.view({log_probs_concat.sizes()[0] * log_probs_concat.sizes()[1], log_probs_concat.sizes()[2]});
			//printf("1\n");
			// std::cout << "Rewards size: " << rewards.sizes() << " " << rewards.size(0) << std::endl;
			//rewards_concat = rewards_concat.reshape({rewards_concat.numel(), 1});
			//dones_concat = dones_concat.reshape({dones_concat.numel(), 1});
			//printf("1\n");
			// advantages_concat = advantages_concat.reshape({advantages_concat.numel()});
			// printf("1\n");
			for(size_t i = 0; i < count_players; i++)
			{
				rewards_concat.insert(rewards_concat.end(), rewards[i].begin(), rewards[i].end());
				dones_concat.insert(dones_concat.end(), dones[i].begin(), dones[i].end());
			}
		}

		//printf("1\n");
		std::vector<float> rewards_concat_ret;
		std::vector<bool> dones_concat_ret;
		torch::Tensor states_concatenated_ret, actions_concat_ret, log_probs_concat_ret;

		size_t start = i * batch_size;
		size_t end = start + batch_size;
		//printf("1\n");
		for(size_t i = end - 1; i > start; i--)
		{
			if(dones_concat[i])
			{
				end = i + 1;
				break;
			}
		}
		//printf("1\n");
		if(start != 0 && dones_concat[start - 1] != true)
		{
			for(size_t i = start; i < end; i++)
			{
				if(dones_concat[i])
				{
					start = i + 1;
					break;
				}
			}
		}
		//std::cout << start << " " << batch_size << std::endl;
		//printf("1\n");
		//dones_concat = dones_concat.reshape({dones_concat.numel(), 1});
		//printf("1\n");
		states_concatenated_ret = states_concatenated_reshaped.index({torch::indexing::Slice(start, end)});
		actions_concat_ret = actions_concat_reshaped.index({torch::indexing::Slice(start, end)});
		log_probs_concat_ret = log_probs_concat_reshaped.index({torch::indexing::Slice(start, end)});
		//rewards_concat_ret = rewards_concat.index({torch::indexing::Slice(start, end)});
		//dones_concat_ret = dones_concat.index({torch::indexing::Slice(start, end)});
		//printf("ended\n");
		//advantages_concat = advantages_concat.index({torch::indexing::Slice(start, end)});
		rewards_concat_ret = std::vector<float>(rewards_concat.begin() + start, rewards_concat.begin() + end);
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
	size_t _capacity;
	torch::Tensor states_concatenated, actions_concat, log_probs_concat;
	torch::Tensor states_concatenated_reshaped, actions_concat_reshaped, log_probs_concat_reshaped;
	std::vector<float> rewards_concat;
	std::vector<bool> dones_concat;
	std::vector<torch::Tensor> /*states,*/ actions, log_probs;
	std::vector<std::vector<float>> rewards;
	std::vector<std::vector<bool>> dones;
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

//auto PPO::returns(VT& rewards, VT& dones, VT& vals, double gamma, double lambda) -> VT
//{
//    // Compute the returns.
//	torch::Tensor gae = torch::zeros({1}, torch::kF32);
//    VT returns(rewards.size(), torch::zeros({1}, torch::kF32));
//
//    for (uint i=rewards.size();i-- >0;) // inverse for loops over unsigned: https://stackoverflow.com/questions/665745/whats-the-best-way-to-do-a-reverse-for-loop-with-an-unsigned-index/665773
//    {
//        // Advantage.
//        auto delta = rewards[i] + gamma*vals[i+1]*(1-dones[i]) - vals[i];
//        gae = delta + gamma*lambda*(1-dones[i])*gae;
//
//        returns[i] = gae + vals[i];
//    }
//
//    return returns;
//}

torch::Tensor compute_advantages(ActorCritic &ac, const torch::Tensor &returns, const torch::Tensor &values)
{
	//auto values = ac->critic_forward(states);
	//std::cout << values.sizes() << std::endl;
	return returns - values;
}

std::vector<float> tensor_to_vector(const torch::Tensor &tensor)
{
	auto tensor_cpu = tensor.to(torch::kCPU, true);
	at::cuda::stream_synchronize(at::cuda::getCurrentCUDAStream());

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

torch::Tensor calculate_returns(std::vector<float> &rewards, std::vector<bool> &dones, const torch::Tensor &values, float gamma, float lambda)
{
	//printf("FINNNN1.1\n");

	float gae = 0;
	std::vector<float> returns(rewards.size());
	std::vector<float> vValues = tensor_to_vector(values);

	for(int64_t i = rewards.size() - 1; i >= 0; --i)
	{
		float delta = 0;
		if(i == rewards.size() - 1)
			delta = (rewards[i] / 200.f) + gamma * vValues[i] * (1 - dones[i]) - vValues[i];
		else
			delta = (rewards[i] / 200.f) + gamma * vValues[i + 1] * (1 - dones[i]) - vValues[i];

		gae = delta + gamma * lambda * (1 - dones[i]) * gae;
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

// Optimized implementation using tensor operations
//torch::Tensor calculate_returns(torch::Tensor &rewards, torch::Tensor &dones, float gamma)
//{
//	auto options = rewards.options();
//	auto device = rewards.device();
//	auto decide_time = std::chrono::high_resolution_clock::now();
//	printf("1\n");
//
//	rewards = rewards.reshape({rewards.numel()});
//	dones = dones.reshape({dones.numel()});
//
//	torch::Tensor rewards_flipped = rewards.flip(0);
//	std::cout << rewards_flipped.sizes() << std::endl;
//
//	printf("1\n");
//
//	torch::Tensor dones_flipped = dones.to(torch::kFloat32).flip(0);
//	std::cout << dones_flipped.sizes() << std::endl;
//
//	printf("1\n");
//
//
//	torch::Tensor discounted = rewards_flipped * torch::pow(gamma, torch::arange(rewards_flipped.size(0), options).to(device));
//	std::cout << discounted.sizes() << std::endl;
//
//	printf("1\n");
//
//	torch::Tensor discounted_returns = discounted.cumsum(0) * torch::pow(gamma, -torch::arange(rewards_flipped.size(0), options).to(device));
//	std::cout << discounted_returns.sizes() << std::endl;
//
//	printf("1\n");
//
//	discounted_returns = discounted_returns.flip(0);
//	std::cout << discounted_returns.sizes() << std::endl;
//
//	printf("1\n");
//
//	torch::Tensor mask = (1.0 - dones_flipped).flip(0);
//	std::cout << mask.sizes() << std::endl;
//
//	printf("1\n");
//
//	discounted_returns = discounted_returns * mask.cumprod(0).flip(0);
//	std::cout << discounted_returns.sizes() << std::endl;
//
//	printf("1\n");
//	auto now = std::chrono::high_resolution_clock::now();
//	std::cout << "Time to calculate_returns: " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(now - decide_time).count()) << std::endl;
//
//
//	return discounted_returns.view({rewards.size(0), 1}).detach();
//}

auto PPO::save_replay(torch::Tensor& state,
    torch::Tensor& action,
    torch::Tensor& log_prob,
    std::vector<float> &reward,
	std::vector<bool> &done) -> void
{
	//torch::NoGradGuard no_grad;
	replay_buffer->add(state, action, log_prob, reward, done);
}

auto PPO::count_of_replays() -> size_t
{
	return replay_buffer->size();
}

auto PPO::update(ActorCritic &ac,
	std::shared_ptr<torch::optim::Adam> &opt,
	uint steps, uint epochs, uint mini_batch_size, double beta, float gamma, float lambda, c10::DeviceType device, double &avg_training_loss, double &avg_actor_loss, double &avg_critic_loss, double clip_param) -> void
{
	torch::Tensor total_loss_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate loss
	torch::Tensor total_actor_loss_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate loss
	torch::Tensor total_critic_loss_tensor = torch::zeros({}, torch::kCUDA); // Initialize tensor to accumulate loss

	{
		std::deque<torch::Tensor> states, actions, values, log_probs;
		std::deque<std::vector<float>> rewards;
		std::deque<std::vector<bool>> dones;
		// Wait for all log probs to come to cpu
		at::cuda::getCurrentCUDAStream().synchronize();
		//at::cuda::stream_synchronize(at::cuda::getCurrentCUDAStream());
		//printf("1\n");
		//printf("1\n");
		for(size_t i = 0; i < replay_buffer->capacity() / mini_batch_size /*&&  i < epochs*/; i++)
		{
			//printf("1.0 %llu %llu %d\n", i, replay_buffer->size(), mini_batch_size);
			auto [state, action, log_prob, reward, done] = replay_buffer->sample(mini_batch_size, i);
			states.push_back(state);
			actions.push_back(action);
			log_probs.push_back(log_prob);
			rewards.push_back(reward);
			dones.push_back(done);
			//printf("1.1\n");

			//state = state.to(device, true);
			//printf("1.2\n");
			{
				//printf("1.2.1\n");
				torch::Tensor cpy_inputs = state.index({"...", torch::indexing::Slice(0, 78)});
				//printf("UPDATING0.3\n");
				//std::cout << state.sizes() << std::endl;
				torch::Tensor cpy_blocks = torch::one_hot(state.index({"...", torch::indexing::Slice(78, 1167)}).to(torch::kInt64), 3).to(torch::kF32).view({(long long)mini_batch_size, -1});
				//printf("UPDATING0.4\n");
				auto cpy_state_forward = torch::cat({cpy_inputs, cpy_blocks}, 1);
				//printf("UPDATING0.5\n");
				//std::cout << cpy_inputs.sizes() << std::endl;
				//std::cout << cpy_blocks.sizes() << std::endl;
				try
				{
					values.push_back(ac->critic_forward(cpy_state_forward).detach());
				}
				catch(const std::exception &e)
				{
					std::cout << e.what() << std::endl;
				}
				//printf("UPDATING0.6\n");
			}
			
		}
		//std::cout << replay_buffer->size() << std::endl;
		//std::cout << mini_batch_size << std::endl;
		//std::cout << replay_buffer->size() / mini_batch_size << std::endl;
		//printf("2\n");
		//Sleep(5000);
		replay_buffer->clear();
		
		//printf("CHECK\n");
		// Sleep(10000);
		// printf("CLEARING\n");
		// torch::cuda::synchronize();
		// Sleep(10000);
		//c10::cuda::CUDACachingAllocator::emptyCache();


		for(uint e = 0; e < epochs; e++)
		{
			for(size_t i = 0; i < replay_buffer->capacity() / mini_batch_size; i++)
			{
				//c10::cuda::CUDACachingAllocator::emptyCache();
				//auto decide_time = std::chrono::high_resolution_clock::now();

				torch::Tensor states_cpy = states[i];
				torch::Tensor actions_cpy = actions[i];
				torch::Tensor log_probs_cpy = log_probs[i].detach();
				
				torch::Tensor cpy_values = values[i];
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
				//printf("UPDATING0.2\n");
				
				torch::Tensor cpy_inputs = cpy_sta.index({"...", torch::indexing::Slice(0, 78)});
				// printf("UPDATING0.3\n");
				torch::Tensor cpy_blocks = torch::one_hot(cpy_sta.index({"...", torch::indexing::Slice(78, 1167)}).to(torch::kInt64), 3).to(torch::kF32).view({(long long)mini_batch_size, -1});
				// printf("UPDATING0.4\n");
				cpy_sta = torch::cat({cpy_inputs, cpy_blocks}, 1);

				// std::cout << cpy_sta.sizes() << std::endl;
				// printf("UPDATING0.1.1\n");
				torch::Tensor cpy_act = actions_cpy;
				// printf("UPDATING0.1.2\n");
				torch::Tensor cpy_log = log_probs_cpy;
				// printf("UPDATING0.1.3\n");
				//  auto catted = torch::cat(rewards).reshape({mini_batch_size, 1});
				//  printf("UPDATING0.1.3.1\n");
				//  std::cout << catted.sizes() << std::endl;
				// printf("UPDATING0.1.3.2\n");
				// std::cout << dones_cpy.sizes() << std::endl;
				// std::cout << dones_cpy << std::endl;
				//printf("3\n");
				//Sleep(7000);
				//std::cout << cpy_values << std::endl;
				auto returnsee = calculate_returns(rewards[i], dones[i], cpy_values, gamma, lambda);
				//auto now = std::chrono::high_resolution_clock::now();
				//std::cout << "Time to prepare: " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(now - decide_time).count()) << std::endl;
				
				// std::cout << returnsee.sizes() << std::endl;
				//std::cout << returnsee << std::endl;
				// auto decide_time = std::chrono::high_resolution_clock::now();

				torch::Tensor cpy_ret = returnsee; // normalize_rewards(returnsee);
				// std::cout << cpy_ret << std::endl;

				// printf("UPDATING0.1.4\n");
				// printf("UPDATING0.1.5\n");
				// std::cout << val.sizes() << std::endl;
				torch::Tensor cpy_adv = compute_advantages(ac, cpy_ret, cpy_values /*cpy_sta.view({cpy_sta.size(0), 1, 4372})*/);
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
				auto action = ac->actor_forward(cpy_sta);
				//printf("4\n");
				//Sleep(7000);
				// printf("33.0\n");
				// std::cout << action.sizes() << std::endl;
				// std::cout << cpy_act.sizes() << std::endl;
				// auto bb = ac->normal_actor(action);
				// auto av = ac->forward(cpy_sta); // action value pairs
				// printf("UPDATING1.2\n");
				// auto action = std::get<0>(av);
				auto entropy = ac->entropy().mean();
				// printf("UPDATING1.3\n");
				auto new_log_prob = ac->log_prob(cpy_act);
				// printf("UPDATING1.4\n");
				auto old_log_prob = cpy_log;
				// printf("UPDATING1.4.1\n");
				//  std::cout << new_log_prob.sizes() << " " << old_log_prob.sizes() << std::endl;
				auto ratio = (new_log_prob - old_log_prob).exp();
				// printf("UPDATING1.5\n");
				//  std::cout << ratio.sizes() << std::endl;
				//  std::cout << cpy_adv.sizes() << std::endl;
				auto surr1 = ratio * cpy_adv;
				// printf("UPDATING1.5.1\n");
				auto surr2 = torch::clamp(ratio, 1. - clip_param, 1. + clip_param) * cpy_adv;
				// printf("UPDATING1.6\n");
				//printf("4.9\n");
				//Sleep(7000);
				auto val = ac->critic_forward(cpy_sta);
				//printf("5\n");
				//Sleep(7000);
				auto actor_loss = -torch::min(surr1, surr2).mean();
				// printf("UPDATING1.7\n");
				auto critic_loss = torch::nn::functional::mse_loss(val, cpy_ret); //(cpy_ret - val).pow(2).mean();
				// printf("UPDATING1.8\n");
				auto loss = 0.5 * critic_loss + actor_loss - beta * entropy;

				// printf("UPDATING1.9\n");
				//  std::cout << "Actor Loss: " << actor_loss.item<double>() << ", Critic Loss: " << critic_loss.item<double>() << std::endl;

				opt->zero_grad();
				// printf("UPDATING1.10\n");
				try
				{
					loss.backward();
				}
				catch(const std::exception &e)
				{
					std::cerr << "Exception during backward pass: " << e.what() << std::endl;
				}
				// torch::nn::utils::clip_grad_norm_(ac->parameters(), 1.0); // Clip gradients
				// printf("UPDATING1.11\n");
				opt->step();
				// bb = ac->normal_actor(action);

				// printf("UPDATING1.12\n");
				// total_loss += loss.item<double>();
				total_actor_loss_tensor += actor_loss;
				total_critic_loss_tensor += critic_loss;
				total_loss_tensor += loss;
				//printf("Pre next\n");
				//Sleep(5000);

				// printf("Chillin\n");
				// Sleep(10000);
				/*states.erase(states.begin());
				actions.erase(actions.begin());
				values.erase(values.begin());
				log_probs.erase(log_probs.begin());
				rewards.erase(rewards.begin());
				dones.erase(dones.begin());*/
				// auto now = std::chrono::high_resolution_clock::now();
				// std::cout << "Time to prepare: " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(now - decide_time).count()) << std::endl;
			}
		}

		
	}
	double avg_loss = 0;
	//auto decide_time = std::chrono::high_resolution_clock::now();

	avg_training_loss = total_loss_tensor.item<double>() / (epochs * replay_buffer->capacity() / mini_batch_size);
	avg_actor_loss = total_actor_loss_tensor.item<double>() / (epochs * replay_buffer->capacity() / mini_batch_size);
	avg_critic_loss = total_critic_loss_tensor.item<double>() / (epochs * replay_buffer->capacity() / mini_batch_size);
	//auto now = std::chrono::high_resolution_clock::now();
	//std::cout << "Time to calculate loss: " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(now - decide_time).count()) << std::endl;
	//std::cout << "Average training Loss: " << avg_loss << std::endl;

	//c10::cuda::CUDACachingAllocator::emptyCache();

	return;
}
