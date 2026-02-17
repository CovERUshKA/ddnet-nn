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
#include <ATen/autocast_mode.h>
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
int64_t h_lstm = 256; // 256
int64_t seq_len = 32;
int64_t lstm_layers = 1; // 1024 256
double std_dev = 0.37; // log(0.37) ~ -1
double learning_rate = 5e-5;
double actor_learning_rate = 3e-4; // 3e-4
double log_std_learning_rate = 1e-5; // 1e-4 - global, 1e-5 state-dependent
double critic_learning_rate = 5e-4; // 1e-3
double lstm_learning_rate = 1e-4; // 1e-4
//double weight_decay = 0.0001;

int64_t mini_batch_size = 8000; // 8000 is the best I think
int64_t count_mini_batches = 1;
//int64_t max_mini_batch_size = 8000; // 4096, 8192, 16384, 32768
int64_t ppo_epochs = 2; // 4
double ent_coef = 1e-2; // Entropy coefficient
//double min_ent_coef = 5e-3;
//double ent_decay_step = (ent_coef - min_ent_coef) / 300.;
//double ent_decay_factor = 0.95;
double clip_param = 0.2; // Default: 0.2
float gamma = 0.99f; // Default: 0.99f Discount factor
float lambda = 0.95f; // GAE lambda

float old_models_train = 0.2f; // Percent of old models. 0.2 = 0.25% of old models
int count_cached_old_models = 100; // old_models_train * ((float)count_bots / 2.f)

int warmup_index = 0;

ActorCritic ac_update;
ActorCritic ac_work;
std::shared_ptr<torch::optim::Adam> opt;
//std::shared_ptr<torch::optim::ReduceLROnPlateauScheduler> scheduler;

std::deque<ActorCritic> old_ac;
std::vector<int> old_bots_indexes;
std::vector<int> input_to_model_id;
torch::Tensor input_to_model_id_tensor_old_indexes;
torch::Tensor input_to_model_id_tensor_current_indexes;
bool graph_recorded = false;

// Tested CUDA Graphs and it produced numerical instability with XMP profile turned on. Turn XMP profile off and make sure the system is stable.
// Graph old model tensors
torch::Tensor graph_input_tensors,
			graph_h_input_tensors,
			graph_c_input_tensors,
			graph_output_tensors,
			graph_h_output_tensors,
			graph_c_output_tensors;
// Graph current model tensors
torch::Tensor graph_main_input_tensor,
			graph_h_main_input_tensor,
			graph_c_main_input_tensor,
			graph_main_output_tensor,
			graph_h_main_output_tensor,
			graph_c_main_output_tensor;
torch::Tensor graph_categorical_rand,
	      graph_normal_rands,
	      graph_bernoulli_rands;
torch::Tensor graph_actions_original,
			graph_actions_sampled;
torch::Tensor graph_old_current_sampled_tensor, graph_current_sampled_tensor;
torch::Tensor graph_log_probs_tensor;
at::cuda::CUDAGraph graph;
at::cuda::CUDAStream graph_stream = at::cuda::getStreamFromPool();

VT states;
VT actions;
VT log_probs;
torch::Tensor h_lstm_states_saved;
torch::Tensor c_lstm_states_saved;
std::vector<float> rewards;
std::vector<bool> dones;
std::vector<bool> accumulation_resets;

// Cache LSTM states
torch::Tensor h_lstm_states;
torch::Tensor c_lstm_states;

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

void ModelManager::ResetCUDAGraph()
{
	if(!input_to_model_id.empty())
	{
		// Convert input_to_model_id to a tensor for use in the graph
		auto options = torch::TensorOptions()
				       .dtype(torch::kInt32);
		torch::Tensor input_to_model_id_tensor = torch::from_blob(input_to_model_id.data(), {static_cast<int>(input_to_model_id.size())}, options).to(device, true);
		input_to_model_id_tensor_current_indexes = (input_to_model_id_tensor == -1).nonzero().flatten();
		input_to_model_id_tensor_old_indexes = (input_to_model_id_tensor != -1).nonzero().flatten();
	}
	else
	{
		input_to_model_id_tensor_current_indexes = torch::arange(0, count_bots, torch::kInt64).to(torch::kCUDA, true);
		input_to_model_id_tensor_old_indexes = torch::empty({0}, torch::kInt64).to(torch::kCUDA, true);
	}
	// std::cout << input_to_model_id_tensor_current_indexes << std::endl;
	// std::cout << input_to_model_id_tensor_old_indexes << std::endl;

	if(is_training)
	{
		int old_models_count = (int)old_bots_indexes.size();
		graph_input_tensors = torch::empty({old_models_count, n_in}, torch::kCUDA);
		graph_h_input_tensors = torch::zeros({lstm_layers, old_models_count, h_lstm}, torch::kCUDA);
		graph_c_input_tensors = torch::zeros({lstm_layers, old_models_count, h_lstm}, torch::kCUDA);
		graph_output_tensors = torch::empty({old_models_count, ac_work->n_out}, torch::kCUDA);
		graph_h_output_tensors = torch::zeros({lstm_layers, old_models_count, h_lstm}, torch::kCUDA);
		graph_c_output_tensors = torch::zeros({lstm_layers, old_models_count, h_lstm}, torch::kCUDA);

		// Old sampled actions
		graph_old_current_sampled_tensor = torch::zeros({old_models_count, 5}, torch::kCUDA);
	}
	int current_models_count = count_bots - old_bots_indexes.size();
	graph_main_input_tensor = torch::empty({current_models_count, n_in}, torch::kCUDA);
	graph_h_main_input_tensor = torch::zeros({lstm_layers, current_models_count, h_lstm}, torch::kCUDA);
	graph_c_main_input_tensor = torch::zeros({lstm_layers, current_models_count, h_lstm}, torch::kCUDA);
	// Output Tensors
	graph_main_output_tensor = torch::empty({current_models_count, ac_work->n_out}, torch::kCUDA);
	graph_h_main_output_tensor = torch::zeros({lstm_layers, current_models_count, h_lstm}, torch::kCUDA);
	graph_c_main_output_tensor = torch::zeros({lstm_layers, current_models_count, h_lstm}, torch::kCUDA);

	// Current model sampling actions
	graph_categorical_rand = torch::zeros({current_models_count, 3}, torch::kCUDA);
	graph_normal_rands = torch::zeros({current_models_count, 2}, torch::kCUDA);
	graph_bernoulli_rands = torch::zeros({2, current_models_count, 1}, torch::kCUDA);
	graph_current_sampled_tensor = torch::zeros({current_models_count, 5}, torch::kCUDA);

	graph_actions_original = torch::zeros({count_bots, ac_work->n_out}, device);
	graph_actions_sampled = torch::zeros({count_bots, 5}, device);

	graph_log_probs_tensor = torch::zeros({count_bots, 1}, device);

	graph_recorded = false;
	graph.reset();
	// Forcefully clear the CUDA cache to free up cached memory from previous graphs
	c10::cuda::CUDACachingAllocator::emptyCache();
	warmup_index = 0;
}

ModelManager::ModelManager(bool is_training, std::string train_folder, size_t batch_size, size_t count_players, uint64_t seed) :
	batch_size(batch_size), iReplaysPerBot(batch_size / count_players), count_bots(count_players)
{
	this->is_training = is_training;
	this->train_folder = train_folder;

	printf("CUDA is available: %d\n", torch::cuda::is_available());

	torch::manual_seed(seed);

	ac_update->Initialize(n_in, n_out, h_start, h_lstm, lstm_layers, std_dev);
	ac_work->Initialize(n_in, n_out, h_start, h_lstm, lstm_layers, std_dev);

	h_lstm_states_saved = torch::zeros({lstm_layers, count_bots, h_lstm}, torch::kCUDA);
	c_lstm_states_saved = torch::zeros({lstm_layers, count_bots, h_lstm}, torch::kCUDA);

	ResetAllBotsMemory();
	ResetCUDAGraph();

	std::cout << "Autocast dtype: " << torch::autocast::get_autocast_dtype(torch::kCUDA) << std::endl;
	std::cout << "Is Autocast cache enabled: " << torch::autocast::is_autocast_cache_enabled() << std::endl;

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
	if(is_training)
	{
		std::vector<torch::optim::OptimizerParamGroup> param_groups;

		param_groups.push_back(torch::optim::OptimizerParamGroup({ac_update->lstm->parameters()},
			std::make_unique<torch::optim::AdamOptions>(lstm_learning_rate)));
		param_groups.push_back(torch::optim::OptimizerParamGroup({ac_update->actor_network->parameters()},
			std::make_unique<torch::optim::AdamOptions>(actor_learning_rate)));
		param_groups.push_back(torch::optim::OptimizerParamGroup({ac_update->actor_head->parameters()},
			std::make_unique<torch::optim::AdamOptions>(actor_learning_rate)));
		param_groups.push_back(torch::optim::OptimizerParamGroup({ac_update->log_std_head->parameters()},
			std::make_unique<torch::optim::AdamOptions>(log_std_learning_rate)));
		param_groups.push_back(torch::optim::OptimizerParamGroup({ac_update->critic_network->parameters()},
			std::make_unique<torch::optim::AdamOptions>(critic_learning_rate)));

		opt = std::make_shared<torch::optim::Adam>(param_groups);
		// scheduler = std::make_shared<torch::optim::ReduceLROnPlateauScheduler>(*opt, /* mode */ torch::optim::ReduceLROnPlateauScheduler::max, /* factor */ 0.5, /* patience */ 10);
		opt->param_groups()[0].options().set_lr(lstm_learning_rate);
		opt->param_groups()[1].options().set_lr(actor_learning_rate);
		opt->param_groups()[2].options().set_lr(actor_learning_rate);
		opt->param_groups()[3].options().set_lr(log_std_learning_rate);
		opt->param_groups()[4].options().set_lr(critic_learning_rate);
	}

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
		system("PAUSE");
		exit(1);
	}
	printf("Copied.\n");
	if(ac_update->is_training())
	{
		PPO::Initilize(batch_size, count_bots, n_in, lstm_layers, h_lstm, seq_len);
		//int botes = count_bots - old_bots_indexes.size();
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

bool ModelManager::ResetBotMemory(int bot_id)
{

	h_lstm_states[0][bot_id].copy_(torch::zeros({h_lstm}, torch::kCUDA));
	c_lstm_states[0][bot_id].copy_(torch::zeros({h_lstm}, torch::kCUDA));

	return true;
}

bool ModelManager::ResetAllBotsMemory()
{
	h_lstm_states = torch::zeros({lstm_layers, count_bots, h_lstm}, torch::kCUDA);
	c_lstm_states = torch::zeros({lstm_layers, count_bots, h_lstm}, torch::kCUDA);

	return true;
}

bool ModelManager::LoadModels(std::string folder_path, std::string main_model_name, bool load_previous)
{
	if(!fs::exists(folder_path))
	{
		std::cerr << "The folder '" << folder_path << "' does not exist." << std::endl;
		return false;
	}

	torch::load(ac_update, folder_path + "\\models\\" + main_model_name + "_model.pt");
	if(is_training)
	{
		torch::load(*opt, folder_path + "\\models\\" + main_model_name + "_optimizer.pt");
	}

	std::cout << "Main model loaded path: " << folder_path + "\\models\\" + main_model_name + "_model.pt" << std::endl;

	if(is_training && load_previous)
	{
		std::string previous_models_folder = folder_path + "\\models\\previous";
		std::string new_models_folder = train_folder + "\\models\\previous";

		// Check if the folder exists
		if(!fs::exists(previous_models_folder))
		{
			std::cerr << "The folder '" << previous_models_folder << "' does not exist." << std::endl;
			return false;
		}

		// Iterate over all files in the folder
		for(const auto &entry : fs::directory_iterator(previous_models_folder))
		{
			std::string model_filename = entry.path().filename().string();
			std::string model_path = previous_models_folder + "\\" + model_filename;
			std::string new_model_path = new_models_folder + "\\" + model_filename;
			// Check if the file is a regular file and has a .pt extension
			if(entry.is_regular_file() && entry.path().extension() == ".pt")
			{
				fs::copy_file(model_path, new_model_path);
			}
		}

		ReloadCachedModels();
	}

	if(is_training)
	{
		opt->param_groups()[0].options().set_lr(lstm_learning_rate);
		opt->param_groups()[1].options().set_lr(actor_learning_rate);
		opt->param_groups()[2].options().set_lr(actor_learning_rate);
		opt->param_groups()[3].options().set_lr(log_std_learning_rate);
		opt->param_groups()[4].options().set_lr(critic_learning_rate);
	}

	try
	{
		ac_work->copy_from(ac_update.get());
		//*opt_work = *opt_update->load(;
	}
	catch(const std::exception &e)
	{
		std::cout << "ac_work->copy_from crashed with reason: " << e.what() << std::endl;
		system("PAUSE");
		exit(1);
	}

	return true;
}

bool ModelManager::ReloadCachedModels()
{
	if(is_training)
	{
		std::string previous_models_folder = train_folder + "\\models\\previous";

		// Check if the folder exists
		if(!fs::exists(previous_models_folder))
		{
			std::cerr << "The folder '" << previous_models_folder << "' does not exist." << std::endl;
			system("PAUSE");
			exit(1);
			return false;
		}

		old_ac.clear();

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
		//std::sort(model_files.begin(), model_files.end(), compare_by_modification_time);

		// Randomly shuffle model files
		// TODO: Implement seed control for reproducibility
		std::random_device rd;
		std::mt19937 gen(rd());
		std::shuffle(model_files.begin(), model_files.end(), gen);

		// Iterate over all files in the folder
		for(const auto &entry : model_files)
		{
			std::string model_filename = entry.path().filename().string();
			std::string model_path = previous_models_folder + "\\" + model_filename;

			ActorCritic old_model;
			old_model->Initialize(n_in, n_out, h_start, h_lstm, lstm_layers, std_dev);
			torch::load(old_model, model_path);
			old_model->eval();
			old_model->to(device);
			old_ac.push_back(old_model);
			if(old_ac.size() == count_cached_old_models)
			{
				break;
			}
		}

		if(!old_ac.empty())
		{
			ReassignOldModels();
		}

		std::cout << "Number of old models reloaded: " << old_ac.size() << std::endl;
	}

	return true;
}

// Sample from a categorical distribution for a batch
//torch::Tensor sample_categorical_batch(torch::Tensor probs)
//{
//	// Sample using multinomial (1 sample per row)
//	return torch::multinomial(probs, 1, /*replacement=*/true);
//}

torch::Tensor sample_categorical_from_logits(torch::Tensor logits, torch::Tensor rand = torch::Tensor())
{
	const double eps = 1e-20;

	if(!rand.defined())
	{
		rand = torch::rand_like(logits); // Sample random numbers
	}

	auto U = rand.clamp_min(eps);
	auto noise = -torch::log(-torch::log(U));

	return (logits + noise).argmax(1, true);
}


// Sample from a Bernoulli distribution for a batch (boolean output)
torch::Tensor sample_bernoulli_batch(torch::Tensor probs, torch::Tensor rand = torch::Tensor())
{
	if(!rand.defined())
	{
		rand = torch::rand_like(probs); // Sample random numbers
	}
	return (rand < probs); // Return boolean tensor
}

// Helper function for old model batch processing
torch::Tensor process_old_model_batch(torch::Tensor outputs)
{
	// Split tensor components
	//printf("000\n");
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
process_main_network(
	torch::Tensor av_current,
	torch::Tensor categorical_rand = torch::Tensor(),
	torch::Tensor normal_rands = torch::Tensor(),
	torch::Tensor bernoulli_rands = torch::Tensor(),
	bool validating = false
)
{
	torch::Tensor angle_logits = av_current.slice(1, 0, 2);
	torch::Tensor dir_logits = av_current.slice(1, 2, 5);
	torch::Tensor hook_logits = av_current.slice(1, 5, 6);
	torch::Tensor hammer_logits = av_current.slice(1, 6, 7);
	torch::Tensor log_std = av_current.slice(1, 7, 9);

	auto angles = torch::tanh(angle_logits);
	auto dir_probs = torch::softmax(dir_logits, 1);
	auto hooks = torch::sigmoid(hook_logits);
	auto hammers = torch::sigmoid(hammer_logits);

	auto directions =
		(ac_work->is_training() && !validating) ? sample_categorical_from_logits(dir_logits, categorical_rand) : torch::argmax(dir_probs, 1).unsqueeze(1);

	if(ac_work->is_training() && !validating)
	{
		angles = ac_work->fast_normal(angles, log_std, normal_rands);
		if(bernoulli_rands.defined())
		{
			hooks = sample_bernoulli_batch(hooks, bernoulli_rands[0]);
			hammers = sample_bernoulli_batch(hammers, bernoulli_rands[1]);
		}
		else
		{
			hooks = sample_bernoulli_batch(hooks);
			hammers = sample_bernoulli_batch(hammers);
		}
	}
	else
	{
		hooks = hooks > 0.5;
		hammers = hammers > 0.5;
	}

	auto catted = torch::cat({angles,
					 directions.to(torch::kFloat32),
					 hooks.to(torch::kFloat32),
					 hammers.to(torch::kFloat32)},
		1);

	return catted;
}

void autocast_enable()
{
	// Enable autocast for the current scope
	torch::autocast::set_autocast_enabled(device, true);
	//torch::autocast::set_autocast_dtype(device, self.fast_dtype); // #type : ignore[arg - type]
	//torch::autocast::set_autocast_cache_enabled(self._cache_enabled);
}

void autocast_disable()
{
	// Disable autocast for the current scope
	torch::autocast::set_autocast_enabled(device, false);
	torch::autocast::clear_cache();
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
	//cudaError_t err = cudaSuccess;
	auto measure_time = std::chrono::high_resolution_clock::now();

	// Turn off gradient calculation
	torch::NoGradGuard no_grad;
	torch::StreamGuard stream_guard{graph_stream};

	std::vector<ModelOutput> outputs;

	//torch::Tensor old_indexes_cpu = torch::from_blob(old_bots_indexes.data(), {(long long)old_bots_indexes.size()}, torch::kInt32).to(precision);
	torch::Tensor state_cpu = torch::from_blob(input_inputs.data(), {(long long)input_inputs.size(), sizeof(ModelInputInputs) / 4}, torch::kF32).to(precision);
	auto state_gpu = state_cpu.to(device, true);

	// Separate the inputs for old models and the current model
	std::vector<torch::Tensor> old_states;
	std::vector<torch::Tensor> current_states;
	std::vector<size_t> old_indices; // To track the original indices of old model inputs
	std::vector<size_t> current_indices; // To track the original indices of current model inputs
	std::vector<torch::Tensor> old_batches;
	//int graph_counter = 0;
	//int graph_main_counter = 0;
	//printf("C\n");
	if(is_training && old_bots_indexes.size())
	{
		graph_input_tensors.copy_(state_gpu.index_select(0, input_to_model_id_tensor_old_indexes), true);
		//printf("C2\n");
		graph_h_input_tensors.copy_(h_lstm_states.index_select(1, input_to_model_id_tensor_old_indexes), true);
		//printf("C3\n");
		graph_c_input_tensors.copy_(c_lstm_states.index_select(1, input_to_model_id_tensor_old_indexes), true);
	}
	//printf("D\n");

	graph_main_input_tensor.copy_(state_gpu.index_select(0, input_to_model_id_tensor_current_indexes), true);
	graph_h_main_input_tensor.copy_(h_lstm_states.index_select(1, input_to_model_id_tensor_current_indexes), true);
	graph_c_main_input_tensor.copy_(c_lstm_states.index_select(1, input_to_model_id_tensor_current_indexes), true);
	//printf("RR\n");
	for(size_t i = 0; i < input_inputs.size(); ++i)
	{
		if(!input_to_model_id.empty() && input_to_model_id[i] != -1)
		{
			/*graph_input_tensors[graph_counter].copy_(state_gpu[i].reshape({1, n_in}), true);
			graph_h_input_tensors[graph_counter].copy_(h_lstm_states[0][i].reshape({lstm_layers, 1, h_lstm}), true);
			graph_c_input_tensors[graph_counter].copy_(c_lstm_states[0][i].reshape({lstm_layers, 1, h_lstm}), true);*/
			//old_states.push_back(state_gpu[i].reshape({1, n_in}));
			old_indices.push_back(i); // Track the original index
			//graph_counter += 1;
		}
		else
		{
			/*graph_main_input_tensor[graph_main_counter].copy_(state_gpu[i].reshape({n_in}), true);
			graph_h_main_input_tensor[0][graph_main_counter].copy_(h_lstm_states[0][i], true);
			graph_c_main_input_tensor[0][graph_main_counter].copy_(c_lstm_states[0][i], true);*/
			//current_states.push_back(state_gpu[i].reshape({1, n_in}));
			current_indices.push_back(i); // Track the original index
			//graph_main_counter += 1;
		}
	}

	h_lstm_states_saved.copy_(h_lstm_states);
	c_lstm_states_saved.copy_(c_lstm_states);

	// Fill rands
	graph_categorical_rand.copy_(torch::rand_like(graph_categorical_rand, torch::kCUDA));
	graph_normal_rands.copy_(torch::rand_like(graph_normal_rands, torch::kCUDA));
	graph_bernoulli_rands.copy_(torch::rand_like(graph_bernoulli_rands, torch::kCUDA));

	auto now = std::chrono::high_resolution_clock::now();
	time_pre_forward = std::chrono::duration<double>(now - measure_time).count() * 1000.;

	measure_time = std::chrono::high_resolution_clock::now();
	//printf("A\n");
	if(!graph_recorded && warmup_index >= 3)
	{
		graph_stream.synchronize();
		graph.capture_begin();

		// Enable autocast
		autocast_enable();

		auto main_output = ac_work->actor_forward(graph_main_input_tensor, graph_h_main_input_tensor, graph_c_main_input_tensor);
		graph_main_output_tensor.copy_(std::get<0>(main_output), true);
		graph_h_main_output_tensor.copy_(std::get<1>(main_output), true);
		graph_c_main_output_tensor.copy_(std::get<2>(main_output), true);

		// Disable autocast
		autocast_disable();

		for(int i = 0; i < old_bots_indexes.size(); ++i)
		{
			int model_id = input_to_model_id[old_indices[i]]; // get the model id for this input
			auto av_old = old_ac[model_id]->actor_forward(
				graph_input_tensors[i].reshape({1, ac_work->n_in}),
				graph_h_input_tensors[0][i].reshape({lstm_layers, 1, h_lstm}),
				graph_c_input_tensors[0][i].reshape({lstm_layers, 1, h_lstm}));
			graph_output_tensors[i].copy_(std::get<0>(av_old).reshape({ac_work->n_out}), true); // store the result for this old model
			graph_h_output_tensors[0][i].copy_(std::get<1>(av_old).reshape({h_lstm}), true);
			graph_c_output_tensors[0][i].copy_(std::get<2>(av_old).reshape({h_lstm}), true);
		}

		if(old_bots_indexes.size())
		{
			auto old_current_sampled = process_old_model_batch(graph_output_tensors);
			graph_old_current_sampled_tensor.copy_(old_current_sampled, true);
		}

		auto av_current_sampled = process_main_network(graph_main_output_tensor, graph_categorical_rand, graph_normal_rands, graph_bernoulli_rands);
		graph_current_sampled_tensor.copy_(av_current_sampled, true);

		graph_actions_original.index_copy_(0, input_to_model_id_tensor_current_indexes, graph_main_output_tensor);
		graph_actions_sampled.index_copy_(0, input_to_model_id_tensor_current_indexes, graph_current_sampled_tensor);
		h_lstm_states.index_copy_(1, input_to_model_id_tensor_current_indexes, graph_h_main_output_tensor);
		c_lstm_states.index_copy_(1, input_to_model_id_tensor_current_indexes, graph_c_main_output_tensor);
		if(is_training && old_bots_indexes.size())
		{
			graph_actions_original.index_copy_(0, input_to_model_id_tensor_old_indexes, graph_output_tensors);
			graph_actions_sampled.index_copy_(0, input_to_model_id_tensor_old_indexes, graph_old_current_sampled_tensor);
			h_lstm_states.index_copy_(1, input_to_model_id_tensor_old_indexes, graph_h_output_tensors);
			c_lstm_states.index_copy_(1, input_to_model_id_tensor_old_indexes, graph_c_output_tensors);
		}

		if(is_training && !validating)
		{
			auto tLogProbs = ac_work->log_prob(graph_actions_original, graph_actions_sampled);
			graph_log_probs_tensor.copy_(tLogProbs);
		}

		graph.capture_end();
		graph_recorded = true;
	}
	else if(!graph_recorded && warmup_index < 3)
	{
		//printf("RR1\n");
		// Enable autocast
		autocast_enable();
		auto main_output = ac_work->actor_forward(graph_main_input_tensor, graph_h_main_input_tensor, graph_c_main_input_tensor);
		graph_main_output_tensor.copy_(std::get<0>(main_output), true);
		graph_h_main_output_tensor.copy_(std::get<1>(main_output), true);
		graph_c_main_output_tensor.copy_(std::get<2>(main_output), true);
		// Disable autocast
		autocast_disable();
		//printf("RR1.2\n");
		for(int i = 0; i < old_bots_indexes.size(); ++i)
		{
			int model_id = input_to_model_id[old_indices[i]]; // get the model id for this input
			auto av_old = old_ac[model_id]->actor_forward(
				graph_input_tensors[i].reshape({1, ac_work->n_in}),
				graph_h_input_tensors[0][i].reshape({lstm_layers, 1, h_lstm}),
				graph_c_input_tensors[0][i].reshape({lstm_layers, 1, h_lstm}));
			//printf("RR2.2\n");
			graph_output_tensors[i].copy_(std::get<0>(av_old).reshape({ac_work->n_out}), true); // store the result for this old model
			//printf("RR2.3\n");
			graph_h_output_tensors[0][i].copy_(std::get<1>(av_old).reshape({h_lstm}), true);
			//printf("RR2.4\n");
			graph_c_output_tensors[0][i].copy_(std::get<2>(av_old).reshape({h_lstm}), true);
		}
		if(old_bots_indexes.size())
		{
			auto old_current_sampled = process_old_model_batch(graph_output_tensors);
			graph_old_current_sampled_tensor.copy_(old_current_sampled, true);
		}

		auto av_current_sampled = process_main_network(graph_main_output_tensor, graph_categorical_rand, graph_normal_rands, graph_bernoulli_rands);
		graph_current_sampled_tensor.copy_(av_current_sampled, true);

		graph_actions_original.index_copy_(0, input_to_model_id_tensor_current_indexes, graph_main_output_tensor);
		graph_actions_sampled.index_copy_(0, input_to_model_id_tensor_current_indexes, graph_current_sampled_tensor);
		h_lstm_states.index_copy_(1, input_to_model_id_tensor_current_indexes, graph_h_main_output_tensor);
		c_lstm_states.index_copy_(1, input_to_model_id_tensor_current_indexes, graph_c_main_output_tensor);
		if(is_training && old_bots_indexes.size())
		{
			graph_actions_original.index_copy_(0, input_to_model_id_tensor_old_indexes, graph_output_tensors);
			graph_actions_sampled.index_copy_(0, input_to_model_id_tensor_old_indexes, graph_old_current_sampled_tensor);
			h_lstm_states.index_copy_(1, input_to_model_id_tensor_old_indexes, graph_h_output_tensors);
			c_lstm_states.index_copy_(1, input_to_model_id_tensor_old_indexes, graph_c_output_tensors);
		}

		if(is_training && !validating)
		{
			auto tLogProbs = ac_work->log_prob(graph_actions_original, graph_actions_sampled);
			graph_log_probs_tensor.copy_(tLogProbs);
		}

		warmup_index += 1;
	}
	else
	{
		graph.replay();
	}

	now = std::chrono::high_resolution_clock::now();
	time_forward = std::chrono::duration<double>(now - measure_time).count() * 1000.;
	measure_time = std::chrono::high_resolution_clock::now();

	now = std::chrono::high_resolution_clock::now();

	time_normal = std::chrono::duration<double>(now - measure_time).count() * 1000.;
	measure_time = std::chrono::high_resolution_clock::now();
	//printf("RWEWQewqe.2\n");
	//printf("QWEWQEWQEwewqe.2\n");
	// Concatenate all actions into a single tensor
	/*auto tActions_original = torch::cat(all_actions_original, 0).reshape({(int)input_inputs.size(), ac_work->n_out});
	auto tActions_sampled = torch::cat(all_actions_sampled, 0).reshape({(int)input_inputs.size(), 5});*/
	
	now = std::chrono::high_resolution_clock::now();
	time_to_cpu = std::chrono::duration<double>(now - measure_time).count() * 1000.;
	measure_time = std::chrono::high_resolution_clock::now();
	//printf("QRRRRWEWQE\n");

	//using clock = std::chrono::high_resolution_clock;

	//static int count_collected = 0;
	////static int count_collected_overall = 0;
	//static double total_time = 0.0;
	//static double logprob_time = 0.0;
	//static double angle_math_time = 0.0;
	//static double slicing_time = 0.0;
	//static double gpu_to_cpu_time = 0.0;
	//static double cpu_loop_time = 0.0;

	//count_collected++;
	//count_collected_overall++;

	//auto total_start = clock::now();

	if(is_training && !validating)
	{
		//auto t0 = clock::now();
		//logprob_time += std::chrono::duration<double>(clock::now() - t0).count() * 1000.0;
		states.push_back(state_gpu);
		actions.push_back(graph_actions_sampled.clone());
		// values.push_back(tValues);
		log_probs.push_back(graph_log_probs_tensor.clone());
	}
	//auto t0 = clock::now();
	// Process angles
	auto angles = graph_actions_sampled.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)});
	auto ataned = torch::atan2(angles.index({torch::indexing::Slice(), 1}), angles.index({torch::indexing::Slice(), 0}));
	auto angle_x = torch::cos(ataned);
	auto angle_y = torch::sin(ataned);
	//angle_math_time += std::chrono::duration<double>(clock::now() - t0).count() * 1000.0;

	// ================= SLICING =================
	//t0 = clock::now();
	auto directions = graph_actions_sampled.index({torch::indexing::Slice(), torch::indexing::Slice(2, 3)});
	auto hooks = graph_actions_sampled.index({torch::indexing::Slice(), torch::indexing::Slice(3, 4)});
	auto hammers = graph_actions_sampled.index({torch::indexing::Slice(), torch::indexing::Slice(4, 5)});
	//slicing_time += std::chrono::duration<double>(clock::now() - t0).count() * 1000.0;

	// ================= GPU → CPU =================
	//t0 = clock::now();
	angle_x = angle_x.to(torch::kCPU, true);
	angle_y = angle_y.to(torch::kCPU, true);
	directions = (directions.reshape({(int)input_inputs.size()}) - 1).to(torch::kLong).to(torch::kCPU, true);
	hooks = hooks.reshape({(int)input_inputs.size()}).to(torch::kBool).to(torch::kCPU, true);
	hammers = hammers.reshape({(int)input_inputs.size()}).to(torch::kBool).to(torch::kCPU, true);

	// When CPU -> GPU no synchronization needed, but needed when GPU -> CPU https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html
	//cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
	//torch::cuda::synchronize();
	graph_stream.synchronize();

	//gpu_to_cpu_time += std::chrono::duration<double>(clock::now() - t0).count() * 1000.0;
	
	// ================= CPU LOOP =================
	//t0 = clock::now();
	auto angle_x_vec = angle_x.accessor<float, 1>(); // at::Half float
	auto angle_y_vec = angle_y.accessor<float, 1>();
	auto direction_indices_vec = directions.accessor<int64_t, 1>();
	auto hook_indices_vec = hooks.accessor<bool, 1>();
	auto hammer_indices_vec = hammers.accessor<bool, 1>();
	for(size_t i = 0; i < input_inputs.size(); ++i)
	{
		ModelOutput output;
		output.angle = {angle_x_vec[i], angle_y_vec[i]};
		output.direction = direction_indices_vec[i];
		output.hook = static_cast<bool>(hook_indices_vec[i]);
		output.hammer = static_cast<bool>(hammer_indices_vec[i]);
		outputs.push_back(output);
	}
	//cpu_loop_time += std::chrono::duration<double>(clock::now() - t0).count() * 1000.0;
	//total_time += std::chrono::duration<double>(clock::now() - total_start).count() * 1000.0;
	now = std::chrono::high_resolution_clock::now();
	// ================= PRINT =================
	//if(count_collected % 200 == 0)
	//{
	//	printf("\n=== Post-process avg over %d ===\n", count_collected);
	//	printf("Total:         %.6f ms\n", total_time / count_collected);
	//	printf("LogProb:       %.6f ms\n", logprob_time / count_collected);
	//	printf("Angle math:    %.6f ms\n", angle_math_time / count_collected);
	//	printf("Slicing:       %.6f ms\n", slicing_time / count_collected);
	//	printf("GPU->CPU:      %.6f ms\n", gpu_to_cpu_time / count_collected);
	//	printf("CPU loop:      %.6f ms\n", cpu_loop_time / count_collected);
	//	printf("================================\n\n");
	//	total_time = 0.0;
	//	logprob_time = 0.0;
	//	angle_math_time = 0.0;
	//	slicing_time = 0.0;
	//	gpu_to_cpu_time = 0.0;
	//	cpu_loop_time = 0.0;
	//	count_collected = 0;
	//	/*if(count_collected_overall % 5000 == 0)
	//	{
	//		printf("Sleeping for 1 sec\n");
	//		Sleep(1000);
	//	}*/
	//}
	time_process_last = std::chrono::duration<double>(now - measure_time).count() * 1000.;
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

			// Reassign to new model after completion
			//for(size_t i = 0; i < dones.size() && old_ac.size(); i++)
			//{
			//	if(dones[i] && input_to_model_id[i] != -1)
			//	{
			//		input_to_model_id[i] = static_cast<int>(round(random_float() * (float)(old_ac.size() - 1))); // Assign to old model
			//	}
			//}
			//printf("2\n");
			//std::cout << states[0].sizes() << std::endl;
			//std::cout << actions[0].sizes() << std::endl;
			//std::cout << log_probs[0].sizes() << std::endl;

			PPO::save_replay(states[0], actions[0], log_probs[0], h_lstm_states_saved, c_lstm_states_saved, rewards, accumulation_resets, dones, mask, is_full);
			//printf("7\n");

		}
		catch(const std::exception &e)
		{
			std::cout << "PPO::save_replay crashed with reason: " << e.what() << std::endl;
			system("PAUSE");
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

	
	ResetCUDAGraph();

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
	NNStats& stats)
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
		old_model->Initialize(n_in, n_out, h_start, h_lstm, lstm_layers, std_dev);
		old_model->copy_from(ac_update.get());
		old_model->eval();
		std::string file_name = to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
		torch::save(ac_update, train_folder + "\\models\\previous\\" + file_name + "_model.pt");
	}

	try
	{
		PPO::update(ac_update, ac_work, opt, rewards.size(), ppo_epochs,
			mini_batch_size, count_mini_batches, ent_coef, gamma, lambda, device,
			stats,
			clip_param);
	}
	catch(const std::exception &e)
	{
		std::cout << "PPO::update crashed with reason: " << e.what() << std::endl;
		system("PAUSE");
		exit(1);
	}

	/*ent_coef -= ent_decay_step;
	ent_coef = std::max(min_ent_coef, ent_coef);*/

	if(cache_model)
	{
		ReloadCachedModels();
	}
	else if(!old_ac.empty())
	{
		ReassignOldModels();
	}

	ResetAllBotsMemory();

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
