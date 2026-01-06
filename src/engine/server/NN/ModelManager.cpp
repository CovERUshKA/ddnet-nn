#include <math.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <base/vmath.h>
// #include <fstream>
#include "Models.h"
#include "ProximalPolicyOptimization.h"
// #include <iostream>
#include <omp.h>
#include "ModelManager.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAGraph.h>
#include <torch/optim/schedulers/reduce_on_plateau_scheduler.h>

//#include <algorithm> // For std::for_each
//#include <execution> // For std::execution::par
//#include <future> // For std::async and std::future
#include <nvToolsExt.h>
#include <filesystem>

namespace fs = std::filesystem;

int64_t n_in = 40;
int64_t n_out = 7;
int64_t h_start = 1024; // 1024 256
double std_dev = 0.37; // log(0.37) ~ -1
double learning_rate = 5e-5;
double actor_learning_rate = 3e-4;
//double log_std_learning_rate = 1e-4;
double critic_learning_rate = 1e-3;
//double weight_decay = 0.0001;

int64_t mini_batch_size = 8000; // 4096, 8192, 16384, 32768
int64_t count_mini_batches = 1;
int64_t max_mini_batch_size = 8000; // 4096, 8192, 16384, 32768
int64_t ppo_epochs = 4;
double ent_coef = 1e-2; // Entropy coefficient
//double min_ent_coef = 2e-3;
//double ent_decay_factor = 0.95;
double clip_param = 0.2; // Default: 0.2
float gamma = 0.99f; // Default: 0.99f Discount factor
float lambda = 0.95f; // GAE lambda

float old_models_train = 0.2f; // Percent of old models
int count_cached_old_models = 100; // old_models_train * ((float)count_bots / 2.f)

ActorCritic ac_update;
ActorCritic ac_work;
std::shared_ptr<torch::optim::Adam> opt;
std::shared_ptr<torch::optim::ReduceLROnPlateauScheduler> scheduler;

std::deque<ActorCritic> old_ac;
std::vector<int> old_bots_indexes;
std::vector<int> input_to_model_id;

VT states;
VT actions;
std::vector<VT> states_bots;
std::vector<VT> actions_bots;
std::vector<float> rewards;
std::vector<bool> dones;
std::vector<bool> accumulation_resets;

VT log_probs;

static auto precision = torch::kF32; // kHalf kF32
static auto device = torch::kCUDA; // kCPU kCUDA

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

	//learning_rate = lr_set[lr_dist(gen)];
	//gamma = gamma_set[gamma_dist(gen)];
	//dbeta = beta_set[beta_dist(gen)];
	//clip_param = clip_set[clip_dist(gen)];
	ppo_epochs = epochs_set[epochs_dist(gen)];
	mini_batch_size = mini_batch_sizes_set[mini_batch_size_dist(gen)];

	return;
}

ModelManager::ModelManager(bool is_training, std::string train_folder, size_t batch_size, size_t count_players, uint64_t seed) :
	batch_size(batch_size), iReplaysPerBot(batch_size / count_players), count_bots(count_players)
{
	this->is_training = is_training;
	this->train_folder = train_folder;

	printf("CUDA is available: %d\n", torch::cuda::is_available());

	torch::manual_seed(seed);

	ac_update->Initialize(n_in, n_out, h_start, std_dev);
	ac_work->Initialize(n_in, n_out, h_start, std_dev);

	// Global Speedups
	// Produce nondetermenistic behavior even on the same gpu
	// Enable optimized cuDNN algorithms, works best with non-fluxuating input size, perfect for RL
	// https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
	at::globalContext().setBenchmarkCuDNN(true);

	//// Use float32 tensor cores on Ampere GPUs, less precision for ~7x speedup
	//// https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
	//at::globalContext().setAllowTF32CuBLAS(true);
	//at::globalContext().setAllowTF32CuDNN(true);

	//// Used FP16 mixed precision
	//// https://pytorch.org/docs/stable/notes/cuda.html#reduced-precision-reduction-in-fp16-gemms
	//at::globalContext().setAllowFP16ReductionCuBLAS(true);

	ac_update->to(precision);

	// Initialize the Adam optimizer with the parameter group
	std::vector<torch::optim::OptimizerParamGroup> param_groups;

	param_groups.push_back(torch::optim::OptimizerParamGroup({ac_update->actor_network->parameters()},
							std::make_unique<torch::optim::AdamOptions>(actor_learning_rate)));
	param_groups.push_back(torch::optim::OptimizerParamGroup({ac_update->critic_network->parameters()},
							std::make_unique<torch::optim::AdamOptions>(critic_learning_rate)));
	//param_groups.push_back(torch::optim::OptimizerParamGroup({ac_update->log_std_},
		//std::make_unique<torch::optim::AdamOptions>(log_std_learning_rate)));

	opt = std::make_shared<torch::optim::Adam>(param_groups);

	scheduler = std::make_shared<torch::optim::ReduceLROnPlateauScheduler>(*opt, /* mode */ torch::optim::ReduceLROnPlateauScheduler::max, /* factor */ 0.5, /* patience */ 10);
	opt->param_groups()[0].options().set_lr(actor_learning_rate);
	opt->param_groups()[1].options().set_lr(critic_learning_rate);

	//for(auto &param_group : opt->param_groups())
	//{
	//	std::cout << param_group.options().get_lr() << std::endl;
	//	if(param_group.options().get_lr() == actor_learning_rate)
	//	{
	//		printf("Setting\n");
	//		param_group.options().set_lr(actor_learning_rate / 3.);
	//		printf("Setted\n");
	//	}

	//	if(param_group.options().get_lr() == critic_learning_rate)
	//	{
	//		printf("Setting\n");
	//		param_group.options().set_lr(critic_learning_rate / 3.);
	//		printf("Setted\n");
	//	}

	//	/*if(param_group.options().get_lr() == log_std_learning_rate)
	//	{
	//		printf("Setting\n");
	//		param_group.options().set_lr(log_std_learning_rate);
	//		printf("Setted\n");
	//	}*/
	//}
	//Sleep(7000);
	ac_update->to(device);
	ac_work->to(device);
	//Sleep(7000);
	// opt(ac->parameters(), 1e-3);
	ac_update->train(is_training);
	printf("Copying...\n");
	try
	{
		ac_work->copy_from(ac_update.get());
		//*opt_work = *opt_update->load(;
	}
	catch(const std::exception &e)
	{
		std::cout << "ac_work->copy_from crashed with reason: " << e.what() << std::endl;
		exit(1);
	}
	printf("Copied.\n");
	if(ac_update->is_training())
	{
		PPO::Initilize(batch_size, count_bots);
		int botes = count_bots - old_bots_indexes.size();
		//ac_work->presample_normal((batch_size / botes) * 1.5, botes);
		cout << "Learning rate: " << learning_rate
		     << " Actor learning rate: " << actor_learning_rate
		     << " Critic learning rate: " << critic_learning_rate
			<< " Gamma: " << gamma
			<< " Lambda: " << lambda
			<< " Entropy coefficient: " << ent_coef
		    << " Standard deviation: " << gamma
			<< " clip_param: " << clip_param
			<< " Epochs: " << ppo_epochs
			<< " Mini batch size: " << mini_batch_size << endl;
		std::cout << "actor_network: " << ac_update->actor_network << std::endl;
		std::cout << "critic_network: " << ac_update->critic_network << std::endl;
	}
	//at::cuda::setCurrentCUDAStream(myStream);
}

//bool DeleteOldestPreviousModel()
//{
//
//}

// Function to compare files by their last modification time
bool compare_by_modification_time(const fs::directory_entry &a, const fs::directory_entry &b)
{
	return fs::last_write_time(a) > fs::last_write_time(b);
}

bool ModelManager::LoadModels(std::string folder_path, std::string main_model_name, bool load_previous)
{
	if(!fs::exists(folder_path))
	{
		std::cerr << "The folder '" << folder_path << "' does not exist." << std::endl;
		return false;
	}

	torch::load(ac_update, folder_path + "\\models\\" + main_model_name + "_model.pt");
	torch::load(*opt, folder_path + "\\models\\" + main_model_name + "_optimizer.pt");

	std::cout << "Main model loaded path: " << folder_path + "\\models\\" + main_model_name + "_model.pt" << std::endl;

	if(load_previous)
	{
		std::string previous_models_folder = folder_path + "\\models\\previous";
		std::string new_models_folder = train_folder + "\\models\\previous";

		// Check if the folder exists
		if(!fs::exists(previous_models_folder))
		{
			std::cerr << "The folder '" << previous_models_folder << "' does not exist." << std::endl;
			return false;
		}

		// Vector to store .pt files
		std::vector<fs::directory_entry> model_files;

		// Iterate over all files in the folder
		for(const auto &entry : fs::directory_iterator(previous_models_folder))
		{
			// Check if the file is a regular file and has a .pt extension
			if(entry.is_regular_file() && entry.path().extension() == ".pt")
			{
				model_files.push_back(entry);
			}
		}

		// Sort files by last modification time (oldest first)
		std::sort(model_files.begin(), model_files.end(), compare_by_modification_time);

		// Iterate over all files in the folder
		for(const auto& entry : model_files)
		{
			// Check if the file is a regular file and has a .pt extension
			if(entry.is_regular_file() && entry.path().extension() == ".pt")
			{
				std::string model_filename = entry.path().filename().string();
				std::string model_path = previous_models_folder + "\\" + model_filename;
				std::string new_model_path = new_models_folder + "\\" + model_filename;

				ActorCritic old_model;
				old_model->Initialize(n_in, n_out, h_start, std_dev);
				torch::load(old_model, model_path);
				old_model->eval();
				//old_model->copy_from(ac_update.get());
				old_model->to(device);
				old_ac.push_back(old_model);
				fs::copy_file(model_path, new_model_path);
				if(old_ac.size() == count_cached_old_models)
				{
					break;
				}
			}
		}

		if(!old_ac.empty())
		{
			ReassignOldModels();
		}

		std::cout << "Number of old models loaded: " << old_ac.size() << std::endl;
	}

	opt->param_groups()[0].options().set_lr(actor_learning_rate);
	opt->param_groups()[1].options().set_lr(critic_learning_rate);

	try
	{
		ac_work->copy_from(ac_update.get());
		//*opt_work = *opt_update->load(;
	}
	catch(const std::exception &e)
	{
		std::cout << "ac_work->copy_from crashed with reason: " << e.what() << std::endl;
		exit(1);
	}

	return true;
}

// Sample from a categorical distribution for a batch
torch::Tensor sample_categorical_batch(torch::Tensor probs)
{
	// Sample using multinomial (1 sample per row)
	return torch::multinomial(probs, 1, /*replacement=*/true);
}

// Sample from a Bernoulli distribution for a batch (boolean output)
torch::Tensor sample_bernoulli_batch(torch::Tensor probs)
{
	auto rand = torch::rand_like(probs); // Sample random numbers
	return (rand < probs); // Return boolean tensor
}

// Helper function for old model batch processing
torch::Tensor process_old_model_batch(torch::Tensor outputs)
{
	// Split tensor components
	auto angle_logits = outputs.slice(1, 0, 2);
	auto dir_logits = outputs.slice(1, 2, 5);
	auto hook_logits = outputs.slice(1, 5, 6);
	auto hammer_logits = outputs.slice(1, 6, 7);
	//printf("111\n");
	// Process deterministically
	auto angles = torch::tanh(angle_logits);
	auto directions = torch::argmax(dir_logits, 1).unsqueeze(1);
	//printf("222\n");

	auto hooks = hook_logits > 0;
	//printf("333\n");

	auto hammers = hammer_logits > 0;
	//printf("444\n");

	//std::cout << directions.sizes() << std::endl;
	//std::cout << hooks.sizes() << std::endl;
	//std::cout << hammers.sizes() << std::endl;

	return torch::cat({angles,
				    directions.to(torch::kFloat32),
				    hooks.to(torch::kFloat32),
				    hammers.to(torch::kFloat32)},
		1);
}

// Helper function for main network processing
torch::Tensor
process_main_network(torch::Tensor av_current, bool validating = false)
{
	torch::Tensor angle_logits = av_current.slice(1, 0, 2);
	torch::Tensor dir_logits = av_current.slice(1, 2, 5);
	torch::Tensor hook_logits = av_current.slice(1, 5, 6);
	torch::Tensor hammer_logits = av_current.slice(1, 6, 7);
	torch::Tensor log_std_logits = av_current.slice(1, 7, 9);

	auto angles = torch::tanh(angle_logits);
	//printf("keke\n");

	// Directions
	auto dir_probs = torch::softmax(dir_logits, 1);
	auto directions = (ac_work->is_training() && !validating) ? sample_categorical_batch(dir_probs) : torch::argmax(dir_probs, 1).unsqueeze(1);

	// Hooks/Hammers
	auto hooks = torch::sigmoid(hook_logits);
	auto hammers = torch::sigmoid(hammer_logits);
	//printf("3131\n");

	if(ac_work->is_training() && !validating)
	{
		// Bound it between lower_bound and upper_bound:
		double lower_bound = -4.0;
		double upper_bound = 0.0;
		auto log_std = lower_bound + (upper_bound - lower_bound) * ((torch::tanh(log_std_logits) + 1) / 2);
		//auto log_std = log_std_logits/*.clamp_max(0)*/;

		angles = ac_work->fast_normal(angles, log_std);
		//std::cout << angles << std::endl;
		hooks = sample_bernoulli_batch(hooks);
		hammers = sample_bernoulli_batch(hammers);
	}
	else
	{
		hooks = hooks > 0.5;
		hammers = hammers > 0.5;
	}

	//printf("111\n");
	/*std::cout << angles.sizes() << std::endl;
	std::cout << directions.sizes() << std::endl;
	std::cout << hooks.sizes() << std::endl;
	std::cout << hammers.sizes() << std::endl;*/


	auto catted = torch::cat({angles, directions.to(torch::kFloat32), hooks.to(torch::kFloat32), hammers.to(torch::kFloat32)}, 1);

	return catted;
}

std::vector<ModelOutput> ModelManager::Decide(
	std::vector<ModelInputInputs> &input_inputs,
	double &time_pre_forward,
	double &time_forward,
	double &time_normal,
	double &time_to_cpu,
	double &time_process_last,
	bool validating)
{
	cudaError_t err = cudaSuccess;
	//nvtxRangePushA("Decide begin");
	auto measure_time = std::chrono::high_resolution_clock::now();
	torch::NoGradGuard no_grad;
	//printf("Deciding...\n");
	std::vector<ModelOutput> outputs;

	//std::cout << input_inputs.size() << std::endl;
	//nvtxRangePushA("Creating state_cpu and gpu");

	//torch::Tensor old_indexes_cpu = torch::from_blob(old_bots_indexes.data(), {(long long)old_bots_indexes.size()}, torch::kInt32).to(precision);
	torch::Tensor state_cpu = torch::from_blob(input_inputs.data(), {(long long)input_inputs.size(), sizeof(ModelInputInputs) / 4}, torch::kF32).to(precision);
	//printf("2123\n");
	//auto old_indexes_gpu = old_indexes_cpu.to(device, true);
	auto state_gpu = state_cpu.to(device, true);
	//nvtxRangePop();

	//printf("111\n");
	// Separate the inputs for old models and the current model
	std::vector<torch::Tensor> old_states;
	std::vector<torch::Tensor> current_states;
	std::vector<size_t> old_indices; // To track the original indices of old model inputs
	std::vector<size_t> current_indices; // To track the original indices of current model inputs
	//printf("333\n");
	std::vector<torch::Tensor> old_batches;
	//printf("444\n");
	//nvtxRangePushA("Redistributing actions");

	for(size_t i = 0; i < input_inputs.size(); ++i)
	{
		if(!input_to_model_id.empty() && input_to_model_id[i] != -1)
		{
			old_states.push_back(state_gpu[i].reshape({1, n_in}));
			old_indices.push_back(i); // Track the original index
		}
		else
		{
			current_states.push_back(state_gpu[i].reshape({1, n_in}));
			current_indices.push_back(i); // Track the original index
		}
	}
	//nvtxRangePop();

	//old_batches.resize(old_indices.size());

	// Vector to hold the futures
	//std::vector<std::future<void>> futures;
	//printf("1\n");
	//// Process each index asynchronously
	//for(size_t i = 0; i < old_indices.size(); ++i)
	//{
	//	// Launch a task asynchronously
	//	futures.push_back(std::async(std::launch::async, [&, i]() {
	//		cudaSetDevice(0);
	//		at::Stream stream = at::cuda::getStreamFromPool(); // Create stream for this thread
	//		at::cuda::CUDAStreamGuard guard(stream); // Guard the stream in this scope
	//		int model_id = input_to_model_id[old_indices[i]]; // Get the model ID for this input
	//		auto av_old = old_ac[model_id]->actor_forward(state_gpu[old_indices[i]]);
	//		old_batches[old_indices[i]] = av_old; // Store the result for this old model
	//	}));
	//}
	//printf("2\n");
	//if(old_indices.size())
	//{
	//	Sleep(1000);
	//	printf("2.1\n");
	//}

	//// Wait for all tasks to complete
	//for(auto &future : futures)
	//{
	//	future.get();
	//}
	//printf("3\n");

	auto now = std::chrono::high_resolution_clock::now();
	time_pre_forward = std::chrono::duration<double>(now - measure_time).count() * 1000.;

	measure_time = std::chrono::high_resolution_clock::now();
	torch::Tensor main_input = torch::cat(current_states, 0);

	torch::Tensor av_current = ac_work->actor_forward(main_input);

	//printf("2.1\n");
	// Step 2: Record the forward pass into the graph
	// Define the kernel parameters for the forward pass using the model
	for(int i = 0; i < old_states.size(); ++i)
	{
		int model_id = input_to_model_id[old_indices[i]]; // get the model id for this input
		auto av_old = old_ac[model_id]->actor_forward(old_states[i]);
		old_batches.push_back(av_old.reshape({1, ac_work->n_out})); // store the result for this old model
	}
	//std::cout << old_batches.size() << std::endl;
	now = std::chrono::high_resolution_clock::now();
	time_forward = std::chrono::duration<double>(now - measure_time).count() * 1000.;
	measure_time = std::chrono::high_resolution_clock::now();
	//nvtxRangePushA("normal_actor");
	torch::Tensor av_current_sampled, old_current_sampled, old_current;
	//printf("1\n");
	if(old_batches.size())
	{
		old_current = torch::cat(old_batches, 0);
		// printf("1.5\n");
		// std::cout << old_current.sizes() << std::endl;
		old_current_sampled = process_old_model_batch(old_current);
	}
	//printf("1.5\n");

	av_current_sampled = process_main_network(av_current);
	//nvtxRangePop();
	//std::cout << av_current[0] << std::endl;
	//printf("2\n");

	now = std::chrono::high_resolution_clock::now();
	time_normal = std::chrono::duration<double>(now - measure_time).count() * 1000.;
	measure_time = std::chrono::high_resolution_clock::now();
	//nvtxRangePushA("after");
	//printf("3\n");

	// Combine results from old models and current model in the correct order
	std::vector<torch::Tensor> all_actions_sampled(input_inputs.size()), all_actions_original(input_inputs.size());
	for(size_t i = 0; i < old_batches.size(); ++i)
	{
		all_actions_sampled[old_indices[i]] = old_current_sampled[i]; // Place old model results in their original positions
		all_actions_original[old_indices[i]] = old_batches[i].reshape({ac_work->n_out});
	}
	for(size_t i = 0; i < current_states.size(); ++i)
	{
		all_actions_sampled[current_indices[i]] = av_current_sampled[i]; // Place current model results in their original positions
		all_actions_original[current_indices[i]] = av_current[i];
	}
	//printf("888\n");
	//nvtxRangePop();
	//nvtxRangePushA("tActions");
	//printf("4\n");
	
	// Concatenate all actions into a single tensor
	auto tActions_original = torch::cat(all_actions_original, 0).reshape({(int)input_inputs.size(), ac_work->n_out});
	//printf("4.5\n");

	auto tActions_sampled = torch::cat(all_actions_sampled, 0).reshape({(int)input_inputs.size(), 5});

	//auto tActions_cpu = tActions.to(torch::kCPU);
	//nvtxRangePop();
	/*static double maxee_max = 0;
	auto maxeee = abs(tActions_cpu.max().item<double>());
	if(maxeee > 10 && maxeee > maxee_max)
	{
		maxee_max = maxeee;
		printf("Big\n");
		std::cout << "New max: " << maxeee << std::endl;
		std::cout << ac_work->log_std_ << std::endl;
		std::cout << av_current_orig << std::endl;
		std::cout << av_current_normaled << std::endl;
		std::cout << tActions_cpu << std::endl;
	}*/

	//printf("5\n");
	
	now = std::chrono::high_resolution_clock::now();
	time_to_cpu = std::chrono::duration<double>(now - measure_time).count() * 1000.;
	measure_time = std::chrono::high_resolution_clock::now();

	//printf("12\n");

	if(is_training && !validating)
	{
		//torch::Tensor sampled = torch::zeros({(int)input_inputs.size(), 5}, torch::kCUDA);
		////printf("13\n");

		//sampled.slice(1, 0, 2).copy_(tActions.slice(1, 0, 2));
		//sampled.slice(1, 2, 3).copy_(directions);
		//sampled.slice(1, 3, 4).copy_(hooks);
		//sampled.slice(1, 4, 5).copy_(hammers);
		//printf("14\n");
		//printf("14.1\n");
		//std::cout << tActions.sizes() << std::endl;
		//std::cout << sampled.sizes() << std::endl;

		auto tLogProbs = ac_work->log_prob(tActions_original, tActions_sampled);
		states.push_back(state_gpu);
		actions.push_back(tActions_sampled);
		// values.push_back(tValues);
		log_probs.push_back(tLogProbs);
	}
	//printf("6\n");

	// Process angles
	auto angles = tActions_sampled.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)});
	auto ataned = torch::atan2(angles.index({torch::indexing::Slice(), 1}), angles.index({torch::indexing::Slice(), 0}));
	auto angle_x = torch::cos(ataned);
	auto angle_y = torch::sin(ataned);

	auto directions = tActions_sampled.index({torch::indexing::Slice(), torch::indexing::Slice(2, 3)});
	auto hooks = tActions_sampled.index({torch::indexing::Slice(), torch::indexing::Slice(3, 4)});
	auto hammers = tActions_sampled.index({torch::indexing::Slice(), torch::indexing::Slice(4, 5)});

	//printf("15\n");
	angle_x = angle_x.to(torch::kCPU, true);
	angle_y = angle_y.to(torch::kCPU, true);
	directions = (directions.reshape({(int)input_inputs.size()}) - 1).to(torch::kLong).to(torch::kCPU, true);
	hooks = hooks.reshape({(int)input_inputs.size()}).to(torch::kBool).to(torch::kCPU, true);
	hammers = hammers.reshape({(int)input_inputs.size()}).to(torch::kBool).to(torch::kCPU, true);

	// When CPU -> GPU no synchronization needed, but needed when GPU -> CPU https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html
	cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
	
	auto angle_x_vec = angle_x.accessor<float, 1>(); // at::Half float
	//printf("16\n");

	auto angle_y_vec = angle_y.accessor<float, 1>();
	//printf("15\n");
	//std::cout << directions << std::endl;
	auto direction_indices_vec = directions.accessor<int64_t, 1>();
	//printf("15\n");

	auto hook_indices_vec = hooks.accessor<bool, 1>();
	//printf("15\n");

	auto hammer_indices_vec = hammers.accessor<bool, 1>();
	//printf("15\n");

	//std::cout << "Time to forward+normal: " << (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(now - decide_time).count()) / (float)std::chrono::nanoseconds(1s).count() << std::endl;
	for(size_t i = 0; i < input_inputs.size(); ++i)
	{
		ModelOutput output;
		output.angle = {angle_x_vec[i], angle_y_vec[i]};
		output.direction = direction_indices_vec[i];
		output.hook = static_cast<bool>(hook_indices_vec[i]);
		output.hammer = static_cast<bool>(hammer_indices_vec[i]);
		outputs.push_back(output);
	}
	now = std::chrono::high_resolution_clock::now();
	time_process_last = std::chrono::duration<double>(now - measure_time).count() * 1000.;
	//nvtxRangePop();
	//printf("Decided\n");
	return outputs;
}

void ModelManager::Reward(float reward, bool reset_accumulation, bool done)
{
	//float don = (float)done;
	if(!ac_work->is_training())
	{
		return;
	}
	rewards.push_back(reward);
	accumulation_resets.push_back(reset_accumulation);
	dones.push_back(done);

	return;
}

void ModelManager::ErasePlayerReplays(int id)
{
	if(!ac_work->is_training())
	{
		return;
	}

	PPO::erase_player_replays(id);

	return;
}

void ModelManager::SaveReplays(bool& is_full)
{
	//printf("SaveReplays\n");
	if(!ac_work->is_training())
	{
		return;
	}

	if(rewards.size())
	{
		try
		{
			//printf("1\n");

			// Create a vector of indices to keep (excluding old_bots_indexes)
			std::vector<bool> mask;
			int bots_indexes_counter = 0;
			for(size_t i = 0; i < rewards.size(); i++)
			{
				if(bots_indexes_counter < old_bots_indexes.size() && i == old_bots_indexes[bots_indexes_counter])
				{
					mask.push_back(0);
					bots_indexes_counter += 1;
				}
				else
				{
					mask.push_back(1);
				}
			}
			for(size_t i = 0; i < dones.size() && old_ac.size(); i++)
			{
				if(dones[i] && input_to_model_id[i] != -1)
				{
					input_to_model_id[i] = static_cast<int>(round(random_float() * (float)(old_ac.size() - 1))); // Assign to old model
				}
			}
			//printf("2\n");
			//std::cout << states[0].sizes() << std::endl;
			//std::cout << actions[0].sizes() << std::endl;
			//std::cout << log_probs[0].sizes() << std::endl;

			PPO::save_replay(states[0], actions[0], log_probs[0], rewards, accumulation_resets, dones, mask, is_full);
			//printf("7\n");

		}
		catch(const std::exception &e)
		{
			std::cout << "PPO::save_replay crashed with reason: " << e.what() << std::endl;
			exit(1);
		}
		
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

bool ModelManager::IsOldModel(int bot_id)
{
	if(old_ac.empty())
	{
		return false;
	}


	return input_to_model_id[bot_id] != -1;
}

void ModelManager::ReassignOldModels()
{
	old_bots_indexes.clear();
	for(size_t team_id = 0; team_id < (int)(old_models_train * (float)count_bots); team_id++)
	{
		float rande = random_float();
		if(rande < 0.5f)
			old_bots_indexes.push_back(team_id * 2);
		else
			old_bots_indexes.push_back(team_id * 2 + 1);
	}

	// Track which inputs belong to old models and which belong to the current model
	input_to_model_id.assign(count_bots, -1); // -1 means current model
	for(size_t i = 0; i < old_bots_indexes.size(); ++i)
	{
		input_to_model_id[old_bots_indexes[i]] = static_cast<int>(round(random_float() * (float)(old_ac.size() - 1))); // Assign to old model
	}

	return;
}

double ModelManager::GetEntropyCoefficient()
{
	return ent_coef;
}

size_t ModelManager::GetCountEpisodes()
{
	return PPO::count_of_episodes();
}

void ModelManager::Update(double avg_reward, bool cache_model, bool &updated,
	double &avg_training_loss, double &avg_actor_loss, double &avg_critic_loss,
	double &avg_entropy, 
	double &avg_actor_grad_norm, double &avg_critic_grad_norm,
	double &avg_actor_weight_norm, double &avg_critic_weight_norm,
	double &avg_actor_activation_mean, double &avg_actor_activation_std,
	double &critic_mean_absolute_error, double &critic_correlation_coefficient,
	double &avg_angle_entropy, double &avg_hook_entropy, double &avg_hammer_entropy, double &avg_direction_entropy)
{
	// Update.
	if(!ac_work->is_training())
	{
		return;
	}
	
	//printf("UPDATING111\n");
	static double episodes_processed = 0;
	//cout << "All: " << PPO::count_of_episodes() << endl;
	int count_replays = PPO::count_of_replays();
	bool is_new_count_mini_batch_size = false;
	
	auto processed = PPO::count_of_episodes() * ((double)(count_replays - count_replays % (mini_batch_size * count_mini_batches)) / (double)count_replays);
	episodes_processed += processed;

	//scheduler->step(avg_reward);

	if(cache_model)
	{
		ActorCritic old_model;
		old_model->Initialize(n_in, n_out, h_start, std_dev);
		old_model->copy_from(ac_update.get());
		old_model->eval();
		old_ac.push_back(old_model);
		std::string file_name = to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
		torch::save(ac_update, train_folder + "\\models\\previous\\" + file_name + "_model.pt");
		if(old_ac.size() > count_cached_old_models)
		{
			old_ac.pop_front();
		}
	}

	try
	{
		PPO::update(ac_update, ac_work, opt, rewards.size(), ppo_epochs,
			mini_batch_size, count_mini_batches, ent_coef, gamma, lambda, device,
			avg_training_loss, avg_actor_loss, avg_critic_loss,
			avg_entropy,
			avg_actor_grad_norm, avg_critic_grad_norm,
			avg_actor_weight_norm, avg_critic_weight_norm,
			avg_actor_activation_mean, avg_actor_activation_std,
			critic_mean_absolute_error, critic_correlation_coefficient,
			avg_angle_entropy, avg_hook_entropy, avg_hammer_entropy, avg_direction_entropy,
			clip_param);
	}
	catch(const std::exception &e)
	{
		std::cout << "PPO::update crashed with reason: " << e.what() << std::endl;
		exit(1);
	}

	/*ent_coef -=  (1e-2 - min_ent_coef) / 300.;
	ent_coef = std::max(min_ent_coef, ent_coef);*/

	if(!old_ac.empty())
	{
		ReassignOldModels();
	}

	//int botes = count_bots - old_bots_indexes.size();
	//ac_work->presample_normal((batch_size / botes) * 1.5, botes);

	ac_work->copy_from(ac_update.get());
	//std::cout << ac_work->is_training() << std::endl;
	//std::cout << old_ac[0]->is_training() << std::endl;
	updated = true;
}

void ModelManager::Save(std::string filename)
{
	torch::save(ac_update, filename + "_model.pt");
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

bool ModelManager::IsTraining()
{
	return ac_work->is_training();
}
