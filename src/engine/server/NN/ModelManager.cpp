#include <math.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <base/vmath.h>
// #include <fstream>
#include "Models.h"
#include "ProximalPolicyOptimization.h"
// #include <iostream>
#include "ModelManager.h"
#include <c10/cuda/CUDAGuard.h>
#include <torch/optim/schedulers/reduce_on_plateau_scheduler.h>

int64_t n_in = 3345; // 78 + 1089 * 3
int64_t n_out = 9;
double stdrt = 2e-2;
double learning_rate = 5e-5; // Default: 1e-3

int64_t mini_batch_size = 8000; // 4096, 8192, 16384, 32768
int64_t ppo_epochs = 2; // Default: 4
double dbeta = 1e-3; // Default: 1e-3
double clip_param = 0.2; // Default: 0.2
float gamma = 0.99f; // Default: 0.99f
float lambda = 0.95f;

ActorCritic ac(n_in, n_out, stdrt);
std::shared_ptr<torch::optim::Adam> opt; //(ac->parameters(), 1e-2);
std::shared_ptr<torch::optim::ReduceLROnPlateauScheduler> scheduler;

VT states;
VT actions;
std::vector<float> rewards;
std::vector<bool> dones;

VT log_probs;
//VT returns;
//VT values;

static auto precision = torch::kF32; // kHalf kF32
//auto precision_dtype = float; // at::Half
static auto device = torch::kCUDA; // kCPU kCUDA
//static at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();

// Function to generate random hyperparameters
void generate_random_hyperparameters()
{
	// Random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	// Predefined set of learning rates
	std::array<double, 6> lr_set = {1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3};
	std::array<int64_t, 3> epochs_set = {2, 4, 8};
	std::array<int64_t, 3> mini_batch_sizes_set = {4096, 8192, 16384};

	std::array<float, 3> gamma_set = {0.9f, 0.99f, 0.999f};
	std::array<double, 3> beta_set = {0.001, 0.01, 0.1};
	std::array<double, 3> clip_set = {0.1, 0.2, 0.3};

	std::uniform_int_distribution<> lr_dist(0, lr_set.size() - 1); // Learning rate range
	std::uniform_int_distribution<> gamma_dist(0, gamma_set.size() - 1); // Gamma range
	std::uniform_int_distribution<> beta_dist(0, beta_set.size() - 1); // Beta range
	std::uniform_int_distribution<> clip_dist(0, clip_set.size() - 1); // Clip parameter range
	std::uniform_int_distribution<> epochs_dist(0, epochs_set.size() - 1); // Epochs range
	std::uniform_int_distribution<> mini_batch_size_dist(0, mini_batch_sizes_set.size() - 1); // Batch size range

	learning_rate = lr_set[lr_dist(gen)];
	//gamma = gamma_set[gamma_dist(gen)];
	//dbeta = beta_set[beta_dist(gen)];
	//clip_param = clip_set[clip_dist(gen)];
	ppo_epochs = epochs_set[epochs_dist(gen)];
	mini_batch_size = mini_batch_sizes_set[mini_batch_size_dist(gen)];

	return;
}

ModelManager::ModelManager(size_t batch_size, size_t count_players) :
	iReplaysPerBot(batch_size / count_players), count_bots(count_players)
{
	printf("CUDA is available: %d\n", torch::cuda::is_available());
	//net_module.eval();
	//torch::set_num_threads(4);
	//torch::set_num_interop_threads(4);
	//generate_random_hyperparameters();
	ac->to(precision);
	//ac->normal(0., stdrt);
	//ac->eval();
	//learning_rate = 2e-5;
	opt = std::make_shared<torch::optim::Adam>(ac->parameters(), learning_rate);
	//torch::load(ac, "train\\1723320877699\\models\\best_model.pt");
	//torch::load(*opt, "train\\1723320877699\\models\\best_optimizer.pt");
	scheduler = std::make_shared<torch::optim::ReduceLROnPlateauScheduler>(*opt, /* mode */ torch::optim::ReduceLROnPlateauScheduler::max, /* factor */ 0.5, /* patience */ 20);
	/*for(auto &param_group : opt->param_groups())
	{
		param_group.options().set_lr(learning_rate);
	}*/
	cout << "Learning rate: " << learning_rate << " Gamma: " << gamma << " Beta: " << dbeta << " clip_param: " << clip_param << " Epochs: " << ppo_epochs << " Mini batch size: " << mini_batch_size << endl;
	//Sleep(7000);
	ac->to(device);
	ac->presample_normal(iReplaysPerBot, count_bots);
	//Sleep(7000);
	// opt(ac->parameters(), 1e-3);
	PPO::Initilize(batch_size, count_bots);
	//at::cuda::setCurrentCUDAStream(myStream);
}

std::vector<ModelOutput> ModelManager::Decide(
	std::vector<ModelInputInputs> &input_inputs,
	std::vector<ModelInputBlocks> &input_blocks,
	double &time_pre_forward,
	double &time_forward,
	double &time_normal,
	double &time_to_cpu,
	double &time_process_last)
{
	auto decide_time = std::chrono::high_resolution_clock::now();
	torch::NoGradGuard no_grad;
	//printf("Deciding...\n");
	std::vector<ModelOutput> outputs;

	//printf("Count: %i\n", (int)input.size());
	torch::Tensor state_inputs_cpu = torch::from_blob(input_inputs.data(), {(long long)input_inputs.size(), sizeof(ModelInputInputs) / 4}, torch::kF32).to(precision);
	torch::Tensor blocks_input_cpu = torch::from_blob(input_blocks.data(), {(long long)input_blocks.size(), sizeof(ModelInputBlocks) / sizeof(long long)}, torch::kInt64);
	// printf("1\n");
	//std::memcpy(state.data_ptr(), &(input), sizeof(input));
	auto blocks_input_gpu = blocks_input_cpu.to(device, true);
	auto state_inputs_gpu = state_inputs_cpu.to(device, true);

	
	
	//printf("1.1\n");
	auto one_hotted_blocks = torch::one_hot(blocks_input_gpu, 3);
	//printf("1.2\n");
	one_hotted_blocks = one_hotted_blocks.to(precision);
	//printf("1.3\n");
	one_hotted_blocks = one_hotted_blocks.view({(long long)input_inputs.size(), -1});
	//printf("1.4\n");
	torch::Tensor state_forward = torch::cat({state_inputs_gpu, one_hotted_blocks}, 1);
	//printf("2\n");
	//states.push_back(state);
	//  Play.
	//cout << state_forward.sizes() << endl;
	at::cuda::getCurrentCUDAStream().synchronize();

	auto now = std::chrono::high_resolution_clock::now();
	time_pre_forward = std::chrono::duration<double>(now - decide_time).count() * 1000.;
	//std::cout << "Time to allocate and transfer: " << std::chrono::duration<double>(now - decide_time).count() << std::endl;
	decide_time = std::chrono::high_resolution_clock::now();
	auto av = ac->actor_forward(state_forward);
	at::cuda::getCurrentCUDAStream().synchronize();

	now = std::chrono::high_resolution_clock::now();
	time_forward = std::chrono::duration<double>(now - decide_time).count() * 1000.;
	decide_time = std::chrono::high_resolution_clock::now();

	//printf("2.1\n");

	av = ac->normal_actor(av);
	at::cuda::getCurrentCUDAStream().synchronize();

	now = std::chrono::high_resolution_clock::now();
	time_normal = std::chrono::duration<double>(now - decide_time).count() * 1000.;
	decide_time = std::chrono::high_resolution_clock::now();
	
	//std::cout << "Equals: " << (at::cuda::getCurrentCUDAStream() == at::cuda::getDefaultCUDAStream()) << std::endl;
	//printf("33.0\n");
	//printf("33.1\n");
	//actions.push_back(std::get<0>(av));
	// cout << "Printing" << endl;
	// cout << torch::argmax(std::get<0>(av)[0]).item<float>() << endl;
	// cout << "End" << endl;
	// printf("33.2\n");
	//values.push_back(std::get<1>(av));
	//log_probs.push_back(ac->log_prob(std::get<0>(av)));
	// printf("33.3\n");

	// float angle = std::get<0>(av)[0][0].item<float>();
	// output.angle = (fmodf(angle, 1.f) + 1.f) / 2.f;

	// float angle = torch::argmax(std::get<0>(av)[0]).item<float>() / 1608.f;
	// output.angle = angle;

	/*int direction = torch::argmax(std::get<0>(av)[0]).item<int>() - 1;
	output.direction = direction;*/

	// Extract and print each predicted value
	/*for(int j = 0; j < output.size(0); ++j)
	{
		std::cout << "Predictions for element " << (i + j) << ": ";
		auto single_output = output[j];
		for(int k = 0; k < single_output.size(0); ++k)
		{
			std::cout << single_output[k].item<float>() << " ";
		}
		std::cout << std::endl;
	}*/
	//auto decide_time = std::chrono::high_resolution_clock::now();
	
	auto tActions = av; // .to(torch::kCUDA)
	//now = std::chrono::high_resolution_clock::now();
	//std::cout << "Time to .to: " << (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(now - decide_time).count()) / (float)std::chrono::nanoseconds(1s).count() << std::endl;
	//printf("pre1\n");

	auto tActions_cpu = av.clone().to(torch::kCPU, true); // tActions.to(torch::kCPU) av
	torch::Tensor state_gpu = torch::cat({state_inputs_gpu, blocks_input_gpu}, 1);
	//auto decide_time = std::chrono::high_resolution_clock::now();
	
	if(ac->is_training())
	{
		auto tLogProbs = ac->log_prob(tActions);
		states.push_back(state_gpu);
		actions.push_back(tActions);
		// values.push_back(tValues);
		log_probs.push_back(tLogProbs);
	}
	
	//tValues = tValues.to(torch::kCPU);
	//auto now = std::chrono::high_resolution_clock::now();
	//std::cout << "Time to .to: " << std::chrono::duration<double>(now - decide_time).count() << std::endl;
	at::cuda::getCurrentCUDAStream().synchronize();
	now = std::chrono::high_resolution_clock::now();
	time_to_cpu = std::chrono::duration<double>(now - decide_time).count() * 1000.;
	decide_time = std::chrono::high_resolution_clock::now();
	//auto decide_time = std::chrono::high_resolution_clock::now();
	
	//printf("1\n");
	// Process angles
	auto angles = tActions_cpu.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)});
	auto ataned = torch::atan2(angles.index({torch::indexing::Slice(), 1}), angles.index({torch::indexing::Slice(), 0}));
	auto angle_x = torch::cos(ataned);
	auto angle_y = torch::sin(ataned);
	//printf("2\n");
	// Process directions
	auto directions = tActions_cpu.index({torch::indexing::Slice(), torch::indexing::Slice(2, 5)});
	auto direction_indices = torch::argmax(directions, 1) - 1;
	//cout << direction_indices << endl;
	//printf("3\n");
	// Process hooks
	auto hooks = tActions_cpu.index({torch::indexing::Slice(), torch::indexing::Slice(5, 7)});
	auto hook_indices = torch::argmax(hooks, 1);
	//printf("4\n");
	auto jumps = tActions_cpu.index({torch::indexing::Slice(), torch::indexing::Slice(7, 9)});
	auto jump_indices = torch::argmax(hooks, 1);
	
	//printf("5\n");
	auto angle_x_vec = angle_x.accessor<float, 1>(); // at::Half float
	//printf("6\n");
	auto angle_y_vec = angle_y.accessor<float, 1>();
	//printf("7\n");
	auto direction_indices_vec = direction_indices.accessor<int64_t, 1>();
	//printf("8\n");
	auto hook_indices_vec = hook_indices.accessor<int64_t, 1>();
	//printf("9\n");
	auto jump_indices_vec = jump_indices.accessor<int64_t, 1>();
	
	//decide_time = std::chrono::steady_clock::now();
	//float time_sum = 0;
	//auto temp_decide_time = std::chrono::steady_clock::now();
	//auto now = std::chrono::high_resolution_clock::now();

	//std::cout << "Time to forward+normal: " << (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(now - decide_time).count()) / (float)std::chrono::nanoseconds(1s).count() << std::endl;
	for(size_t i = 0; i < input_inputs.size(); ++i)
	{
		ModelOutput output;
		// printf("5\n");
		// auto temp_decide_time = std::chrono::steady_clock::now();

		// auto now = std::chrono::steady_clock::now();
		// time_sum += (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(now - temp_decide_time).count()) / (float)std::chrono::nanoseconds(1s).count();
		// printf("6\n");
		// std::cout << angle_x_vec[i] << std::endl;
		output.angle = {angle_x_vec[i], angle_y_vec[i]};
		// printf("7\n");
		output.direction = direction_indices_vec[i];
		// std::cout << output.angle[0] << std::endl;
		// printf("8\n");
		output.hook = static_cast<bool>(hook_indices_vec[i]);
		output.jump = static_cast<bool>(jump_indices_vec[i]);
		// printf("9\n");
		outputs.push_back(output);

		// printf("10\n");
	}
	now = std::chrono::high_resolution_clock::now();
	time_process_last = std::chrono::duration<double>(now - decide_time).count() * 1000.;
	
	//printf("12323\n");
	
	
	//printf("Decided\n");
	//auto now = std::chrono::high_resolution_clock::now();
	//std::cout << "Time to load: " << (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(now - decide_time).count()) / (float)std::chrono::nanoseconds(1s).count() << std::endl;

	//for(size_t i = 0; i < input_inputs.size(); i++)
	//{
	//	ModelOutput output;
	//	//cout << tensor_actions.sizes() << " " << tensor_actions.is_contiguous() << endl;
	//	//torch::Tensor _state = torch::zeros({1, n_in}, torch::kHalf);
	//	//std::memcpy(_state.data_ptr(), state[i].data_ptr(), 1099 * sizeof(float));
	//	if(ac->is_training())
	//	{
	//		auto log_prob = tLogProbs[i].unsqueeze(0);
	//		auto tensor_values = tValues[i].unsqueeze(0);
	//		states.push_back(state[i].unsqueeze(0));
	//		actions.push_back(tActions[i].unsqueeze(0));
	//		values.push_back(tensor_values);
	//		log_probs.push_back(log_prob);
	//	}

	//	auto temp_decide_time = std::chrono::steady_clock::now();
	//	
	//	//cout << log_prob.sizes() << endl;
	//	//cout << state[i].is_contiguous() << " " << state[i].sizes() << " " << _state.is_contiguous() << " " << _state.sizes() << endl;
	//	auto tensor_actions = tActions[i];
	//	// Big network
	//	auto angles = tensor_actions.index({torch::indexing::Slice(0, 2)});
	//	// cout << "Angles size: " << angles.size(0) << endl;
	//	float x = angles[0].item<float>();
	//	float y = angles[1].item<float>();

	//	// Calculate the angle in radians
	//	float angle_radians = std::atan2(y, x);

	//	// Compute the unit vector components
	//	float angle_x = std::cos(angle_radians);
	//	float angle_y = std::sin(angle_radians);
	//	output.angle = {angle_x, angle_y};

	//	auto directions = tensor_actions.index({torch::indexing::Slice(2, 5)});
	//	//cout << "Directions: " << directions << endl;
	//	int direction = torch::argmax(directions).item<int>() - 1;
	//	output.direction = direction;

	//	auto hooks = tensor_actions.index({torch::indexing::Slice(5, 7)});
	//	bool hook = torch::argmax(hooks).item<bool>();
	//	output.hook = hook;

	//	time_sum += (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - temp_decide_time).count()) / (float)std::chrono::nanoseconds(1s).count();

	//	outputs.push_back(output);
	//}

	//cout << "Time to postprocess: " << time_sum << endl;

	//cout << "Time to postprocess: " << (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - decide_time).count()) / (float)std::chrono::nanoseconds(1s).count() << endl;

	// printf("33.4\n");
	// printf("Angle: %f\n", output.angle);

	/*int64_t n_in = 4;
	int64_t n_out = 2;
	double std = 2e-2;*/

	/*ActorCritic ac(n_in, n_out, std);
	ac->to(torch::kF64);
	ac->normal(0., std);
	torch::optim::Adam opt(ac->parameters(), 1e-3);*/

	return outputs;
}

//ModelOutput ModelManager::Decide(ModelInputInputs &input)
//{
//	torch::NoGradGuard no_grad;
//
//	ModelOutput output;
//	
//	//printf("22\n");
//	torch::Tensor state = torch::zeros({1, n_in}, torch::kHalf);
//	//printf("1\n");
//	std::memcpy(state.data_ptr(), &(input), sizeof(input));
//	state = state.to(device);
//	//printf("2\n");
//	states.push_back(state);
//	//printf("33\n");
//	// Play.
//	auto av = ac->forward(state);
//	//printf("33.1\n");
//	actions.push_back(std::get<0>(av));
//	//cout << "Printing" << endl;
//	//cout << torch::argmax(std::get<0>(av)[0]).item<float>() << endl;
//	//cout << "End" << endl;
//	//printf("33.2\n");
//	//values.push_back(std::get<1>(av));
//	log_probs.push_back(ac->log_prob(std::get<0>(av)));
//	//printf("33.3\n");
//	
//	//float angle = std::get<0>(av)[0][0].item<float>();
//	//output.angle = (fmodf(angle, 1.f) + 1.f) / 2.f;
//
//	//float angle = torch::argmax(std::get<0>(av)[0]).item<float>() / 1608.f;
//	//output.angle = angle;
//
//	/*int direction = torch::argmax(std::get<0>(av)[0]).item<int>() - 1;
//	output.direction = direction;*/
//
//	//Big network
//	auto angles = std::get<0>(av)[0].index({torch::indexing::Slice(0, 2)});
//	//cout << "Angles size: " << angles.size(0) << endl;
//	float x = angles[0].item<float>();
//	float y = angles[1].item<float>();
//
//	// Calculate the angle in radians
//	float angle_radians = std::atan2(y, x);
//
//	// Compute the unit vector components
//	float angle_x = std::cos(angle_radians);
//	float angle_y = std::sin(angle_radians);
//	output.angle = {angle_x, angle_y};
//
//	auto directions = std::get<0>(av)[0].index({torch::indexing::Slice(2, 5)});
//	int direction = torch::argmax(directions).item<int>() - 1;
//	output.direction = direction;
//
//	auto hooks = std::get<0>(av)[0].index({torch::indexing::Slice(5, 7)});
//	bool hook = torch::argmax(hooks).item<bool>();
//	output.hook = hook;
//
//	//printf("33.4\n");
//	//printf("Angle: %f\n", output.angle);
//
//	/*int64_t n_in = 4;
//	int64_t n_out = 2;
//	double std = 2e-2;*/
//
//	/*ActorCritic ac(n_in, n_out, std);
//	ac->to(torch::kF64);
//	ac->normal(0., std);
//	torch::optim::Adam opt(ac->parameters(), 1e-3);*/
//
//	return output;
//}

void ModelManager::Reward(float reward, bool done)
{
	//float don = (float)done;
	if(!ac->is_training())
	{
		return;
	}
	//printf("44\n");
	//torch::NoGradGuard no_grad;
	//torch::Tensor treward = torch::full({1, 1}, reward, torch::kF32);
	//torch::Tensor tdone = torch::full({1, 1}, don, torch::kF32);
	//printf("55\n");
	//treward = treward.to(torch::kCPU);
	//tdone = tdone.to(device);
	//printf("66\n");
	//std::memcpy(treward.data_ptr(), &rew, sizeof(rew));
	//std::memcpy(tdone.data_ptr(), &don, sizeof(don));
	// New state.
	rewards.push_back(reward);
	dones.push_back(done);
	//printf("66\n");
	//PPO::save_replay(states[states.size() - 1], actions[actions.size() - 1], log_probs[log_probs.size() - 1], rewards[rewards.size() - 1], rewards[rewards.size() - 1] - values[values.size() - 1]);
	//printf("77\n");
	// avg_reward += rewards[c][0][0].item<double>() / n_iter;

	// episode, agent_x, agent_y, goal_x, goal_y, AGENT=(PLAYING, WON, LOST, RESETTING)
	// out << e << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << std::get<1>(sd) << "\n";

	// if(dones[c][0][0].item<double>() == 1.)
	//{
	//	// Set new goal.
	//	double x_new = double(dist(re));
	//	double y_new = double(dist(re));
	//	env.SetGoal(x_new, y_new);

	//	// Reset the position of the agent.
	//	env.Reset();

	//	// episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
	//	cout << e << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << RESETTING << "\n";
	//}

	// c++;

	return;
}

void ModelManager::SaveReplays()
{
	if(!ac->is_training())
	{
		return;
	}
	
	/*for(size_t i = 0; i < states.size(); i++)
	{
		PPO::save_replay(states[i], actions[i], log_probs[i], rewards, rewards[i] - values[i]);
	}*/
	if(rewards.size())
	{
		//torch::NoGradGuard no_grad;

		//torch::Tensor tRewards = torch::from_blob(rewards.data(), {(long long)rewards.size(), 1}, torch::kF32);
		//torch::Tensor tDones = torch::from_blob(dones.data(), {(long long)dones.size(), 1}, torch::kF32);

		//tRewards = tRewards.to(device, myStream);
		//tDones = tDones.to(device, myStream);
		//std::cout << tDones.sizes() << std::endl;


		PPO::save_replay(states[0], actions[0], log_probs[0], rewards, dones);
	}

	states.clear();
	actions.clear();
	rewards.clear();
	dones.clear();
	
	log_probs.clear();
	//returns.clear();
	//values.clear();

	return;
}

void ModelManager::Update(double avg_reward, double& avg_training_loss)
{
	// Update.
	//printf("Updating the network.\n");
	//printf("1");
	//values.push_back(std::get<1>(ac->forward(states[states.size() - 1])));

	if(!ac->is_training())
	{
		return;
	}
	//printf("saveing.\n");

	//returns = PPO::returns(normalize_rewards(rewards), dones, values, .99, .95);
	//printf("2");
	//torch::Tensor t_log_probs = torch::cat(log_probs).detach();
	////printf("2.1");
	//torch::Tensor t_returns = normalize_rewards(torch::cat(rewards).detach());
	////printf("2.2");
	//torch::Tensor t_values = torch::cat(values).detach();
	////printf("2.3");
	//torch::Tensor t_states = torch::cat(states);
	////printf("2.4");
	//torch::Tensor t_actions = torch::cat(actions);
	////printf("2.5");
	//torch::Tensor t_advantages = t_returns - t_values.slice(0, 0, rewards.size());
	//printf("3");
	//printf("UPDATING111\n");
	try
	{
		avg_training_loss = PPO::update(ac, opt, rewards.size(), ppo_epochs, mini_batch_size, dbeta, gamma, lambda, device, clip_param);
	}
	catch(const std::exception &e)
	{
		std::cout << "PPO::update crashed with reason: " << e.what() << std::endl;
		exit(1);
	}
	scheduler->step(avg_reward);
	ac->presample_normal(iReplaysPerBot, count_bots);
	for(auto &group : opt->param_groups())
	{
		auto lr = group.options().get_lr();
		std::cout << "Current learning rate: " << lr << std::endl;
	}
	//printf("UPDATed\n");
	//printf("4");
	// c = 0;
	//printf("5");
	//printf("7");
}

void ModelManager::Save(std::string filename)
{
	torch::save(ac, filename + "_model.pt");
	torch::save(*opt, filename + "_optimizer.pt");
}

size_t ModelManager::GetCountOfReplays()
{
	return PPO::count_of_replays();
}

double ModelManager::GetLearningRate()
{
	return learning_rate;
}

double ModelManager::GetCurrentLearningRate()
{
	double lr = 0;
	for(auto &group : opt->param_groups())
	{
		lr = group.options().get_lr();
		break;
	}
	return lr;
}

int64_t ModelManager::GetMiniBatchSize()
{
	return mini_batch_size;
}

int64_t ModelManager::GetCountPPOEpochs()
{
	return ppo_epochs;
}
