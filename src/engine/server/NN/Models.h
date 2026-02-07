#pragma once

#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <math.h>
#include <ctime>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/utils.h>

// Add a small epsilon for numerical stability
constexpr float EPSILON = 1e-7;

// Network model for Proximal Policy Optimization on Incy Wincy.
struct ActorCriticImpl : public torch::nn::Module 
{
	int64_t n_in, n_out, h_lstm;

	torch::nn::LSTM lstm = nullptr;

    // Actor.
    torch::nn::Sequential actor_network;

    // Critic.
    torch::nn::Sequential critic_network;

	torch::nn::Linear actor_head = nullptr, log_std_head = nullptr;

	ActorCriticImpl()
	{

	}

    bool Initialize(int64_t n_in, int64_t n_out, int64_t h_start, int64_t h_lstm, int64_t lstm_layers, double std)
    {
	    this->n_in = n_in;
	    this->n_out = n_out + 2;
	    this->h_lstm = h_lstm;
	    lstm = torch::nn::LSTM(torch::nn::LSTMOptions(n_in, h_lstm).num_layers(lstm_layers).batch_first(true));
	    actor_network = torch::nn::Sequential(
		    torch::nn::Linear(h_lstm, h_start),
		    torch::nn::ReLU(),
		    torch::nn::Linear(h_start, h_start/2),
			torch::nn::ReLU(),
		    torch::nn::Linear(h_start / 2, h_start/4),
		    torch::nn::ReLU(),
		    torch::nn::Linear(h_start / 4, h_start/8),
		    torch::nn::ReLU(),
		    torch::nn::Linear(h_start / 8, h_start / 16),
		    torch::nn::ReLU()
		    //torch::nn::Linear(h_start / 16, n_out)
		    //torch::nn::Tanh()
		    );

		actor_head = torch::nn::Linear(h_start / 16, n_out);

		log_std_head = torch::nn::Linear(h_start / 16, 2);

		//mu_ = torch::full(n_out, 0.);
	    //log_std_ = torch::full(2, std::log(std));
		critic_network = torch::nn::Sequential(
		    torch::nn::Linear(h_lstm, h_start),
		    torch::nn::ReLU(),
		    torch::nn::Linear(h_start, h_start / 2),
		    torch::nn::ReLU(),
		    torch::nn::Linear(h_start / 2, h_start / 4),
		    torch::nn::ReLU(),
		    torch::nn::Linear(h_start / 4, h_start / 8),
		    torch::nn::ReLU(),
		    torch::nn::Linear(h_start / 8, h_start / 16),
		    torch::nn::ReLU(),
		    torch::nn::Linear(h_start / 16, 1)
		);
	    //printf("1\n");
		// Get the last layer (final Linear layer)
		//printf("2\n");

		// Access the bias tensor
		auto bias = log_std_head->bias;
	    //printf("3\n");

		if(bias.defined())
		{
			torch::NoGradGuard no_grad;
			//printf("4\n");
			//std::cout << bias.sizes() << std::endl;
 			// Modify only the last two bias values
			// 0.55 is -1 when transformed with tanh and other values
			bias.index_put_({0}, -1);
			//printf("5\n");
			bias.index_put_({1}, -1);
		}

	    //printf("Created from 0\n");

		register_module("lstm", lstm);
	    register_module("actor_network", actor_network);
		register_module("actor_head", actor_head);
	    register_module("log_std_head", log_std_head);
        //register_parameter("log_std", log_std_);
	    register_module("critic_network", critic_network);

		this->lstm->flatten_parameters();
	
		//std::cout << log_std_ << std::endl;
	    return true;
    }

    // Forward pass.
    auto actor_forward(torch::Tensor x) -> torch::Tensor
    {
        // Actor.
	    torch::Tensor action;
	    try
	    {
		    // Expect x shape [batch, n_in]. Convert to sequence length 1 for LSTM.
		    auto seq = x.unsqueeze(1); // [batch, 1, n_in]
		    auto lstm_out_tuple = lstm->forward(seq);
		    auto lstm_out = std::get<0>(lstm_out_tuple); // [batch, 1, lstm_hidden]
		    auto feat = lstm_out.squeeze(1); // [batch, lstm_hidden]
		    auto hidden = actor_network->forward(feat);
		    action = actor_head->forward(hidden);
		    auto log_std = log_std_head->forward(hidden);
		    // Bound it between lower_bound and upper_bound:
		    double lower_bound = -4.0;
		    double upper_bound = 0.0;
		    //log_std = lower_bound + (upper_bound - lower_bound) * ((torch::tanh(log_std) + 1) / 2);
		    log_std = torch::clamp(log_std, lower_bound, upper_bound);
		    action = torch::cat({action, log_std}, 1); // Concatenate action and log_std for output
	    }
	    catch(const std::exception &e)
	    {
		    std::cout << "actor_network->forward crashed with reason: " << e.what() << std::endl;
		    system("PAUSE");
	    }

	    return action;
    }

	// Forward pass for a sequence with provided LSTM hidden state.
    // Input `x` shape: [batch, n_in]
    // `h`/`c` shape: [num_layers, batch, lstm_hidden]
    // Returns tuple: (actions_seq [batch, n_out], h_out, c_out)
    auto actor_forward(torch::Tensor x, torch::Tensor h, torch::Tensor c) -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    {
	    // Run LSTM with provided hidden state
	    auto seq = x.unsqueeze(1);
	    auto lstm_out_tuple = lstm->forward(seq, std::make_tuple(h, c));
	    auto lstm_out = std::get<0>(lstm_out_tuple); // [batch, seq_len, lstm_hidden]
	    auto h_out_tuple = std::get<1>(lstm_out_tuple);
	    auto h_out = std::get<0>(h_out_tuple);
	    auto c_out = std::get<1>(h_out_tuple);
	    // std::cout << "lstm_out sizes: " << lstm_out.sizes() << std::endl;
	    //  Flatten time dimension to apply actor network to each timestep
	    auto batch = lstm_out.size(0);
	    auto seq_len = lstm_out.size(1);
	    auto feat = lstm_out.reshape({batch * seq_len, this->h_lstm});
	    auto hidden = actor_network->forward(feat);
	    auto actions_flat = actor_head->forward(hidden);
	    auto log_std = log_std_head->forward(hidden);
	    // Bound it between lower_bound and upper_bound:
	    double lower_bound = -4.0;
	    double upper_bound = 0.0;
	    // log_std = lower_bound + (upper_bound - lower_bound) * ((torch::tanh(log_std) + 1) / 2);
	    log_std = torch::clamp(log_std, lower_bound, upper_bound);
	    actions_flat = torch::cat({actions_flat, log_std}, 1); // Concatenate action and log_std for output
	    //auto actions = actions_flat.reshape({batch, seq_len, n_out});
	    // std::cout << "actions sizes: " << lstm_out.sizes() << std::endl;
	    return {actions_flat, h_out, c_out};
    }

	// Forward pass for a sequence with provided LSTM hidden state.
    // Input `seq` shape: [batch, seq_len, n_in]
    // `h`/`c` shape: [batch, num_layers, lstm_hidden]
    // Returns tuple: (actions_seq [batch, seq_len, n_out], h_out, c_out)
    auto actor_forward_sequence(torch::Tensor seq, torch::Tensor h, torch::Tensor c) -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    {
	    // Run LSTM with provided hidden state
	    auto lstm_out_tuple = lstm->forward(seq, std::make_tuple(h, c));
	    auto lstm_out = std::get<0>(lstm_out_tuple); // [batch, seq_len, lstm_hidden]
	    auto h_out_tuple = std::get<1>(lstm_out_tuple);
	    auto h_out = std::get<0>(h_out_tuple);
	    auto c_out = std::get<1>(h_out_tuple);
	    //std::cout << "lstm_out sizes: " << lstm_out.sizes() << std::endl;
	    // Flatten time dimension to apply actor network to each timestep
	    auto batch = lstm_out.size(0);
	    auto seq_len = lstm_out.size(1);
	    auto feat = lstm_out.reshape({batch * seq_len, this->h_lstm});
	    auto hidden = actor_network->forward(feat);
	    auto actions_flat = actor_head->forward(hidden);
	    auto log_std = log_std_head->forward(hidden);
	    // Bound it between lower_bound and upper_bound:
	    double lower_bound = -4.0;
	    double upper_bound = 0.0;
	    // log_std = lower_bound + (upper_bound - lower_bound) * ((torch::tanh(log_std) + 1) / 2);
	    log_std = torch::clamp(log_std, lower_bound, upper_bound);
	    actions_flat = torch::cat({actions_flat, log_std}, 1); // Concatenate action and log_std for output
	    auto actions = actions_flat.reshape({batch, seq_len, n_out});
	    //std::cout << "actions sizes: " << lstm_out.sizes() << std::endl;
	    return {actions, h_out, c_out};
    }

    // Forward pass.
    auto critic_forward(torch::Tensor x) -> torch::Tensor
    {
	    // Critic.
	    // Pass through LSTM first the same way as actor
	    auto seq = x.unsqueeze(1);
	    auto lstm_out_tuple = lstm->forward(seq);
	    auto lstm_out = std::get<0>(lstm_out_tuple);
	    auto feat = lstm_out.squeeze(1);
	    torch::Tensor val = critic_network->forward(feat);
	    return val;
    }

	// Critic forward for sequence with provided LSTM hidden state.
    // Input `x` shape: [batch, n_in]
    // Returns tuple: (values_seq [batch, 1], h_out, c_out)
    auto critic_forward(torch::Tensor x, torch::Tensor h, torch::Tensor c) -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    {
		auto seq = x.unsqueeze(1);
	    auto lstm_out_tuple = lstm->forward(seq, std::make_tuple(h, c));
	    auto lstm_out = std::get<0>(lstm_out_tuple); // [batch, seq_len, lstm_hidden]
	    auto h_out_tuple = std::get<1>(lstm_out_tuple);
	    auto h_out = std::get<0>(h_out_tuple);
	    auto c_out = std::get<1>(h_out_tuple);

	    auto batch = lstm_out.size(0);
	    auto seq_len = lstm_out.size(1);
	    auto feat = lstm_out.reshape({batch * seq_len, this->h_lstm});
	    auto vals_flat = critic_network->forward(feat);
	    return {vals_flat, h_out, c_out};
    }

	// Critic forward for sequence with provided LSTM hidden state.
    // Input `seq` shape: [batch, seq_len, n_in]
    // Returns tuple: (values_seq [batch, seq_len, 1], h_out, c_out)
    auto critic_forward_sequence(torch::Tensor seq, torch::Tensor h, torch::Tensor c) -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    {
	    auto lstm_out_tuple = lstm->forward(seq, std::make_tuple(h, c));
	    auto lstm_out = std::get<0>(lstm_out_tuple); // [batch, seq_len, lstm_hidden]
	    auto h_out_tuple = std::get<1>(lstm_out_tuple);
	    auto h_out = std::get<0>(h_out_tuple);
	    auto c_out = std::get<1>(h_out_tuple);

	    auto batch = lstm_out.size(0);
	    auto seq_len = lstm_out.size(1);
	    auto feat = lstm_out.reshape({batch * seq_len, this->h_lstm});
	    auto vals_flat = critic_network->forward(feat);
	    auto vals = vals_flat.reshape({batch, seq_len, 1});
	    return {vals, h_out, c_out};
    }

	// Copy constructor
    ActorCriticImpl(const ActorCriticImpl *other)
    {
	    // Clone the actor network from the other model
	    actor_network = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(other->actor_network->clone());
		// Clone the actor head from the other model
	    actor_head = std::dynamic_pointer_cast<torch::nn::LinearImpl>(other->actor_head->clone());
		// Clone the log_std head from the other model
	    log_std_head = std::dynamic_pointer_cast<torch::nn::LinearImpl>(other->log_std_head->clone());
	    // Clone the critic network from the other model
	    critic_network = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(other->critic_network->clone());
		// Clone the critic network from the other model
	    lstm = std::dynamic_pointer_cast<torch::nn::LSTMImpl>(other->lstm->clone());
	    this->lstm->flatten_parameters();
	    // Copy the log_std_ parameter
	    //log_std_ = other->log_std_.detach().clone();
	    if(!other->is_training())
	    {
		    this->eval();
	    }
	    else
	    {
		    this->train();
	    }
	    //printf("Copied\n");
	    register_module("lstm", lstm);
	    register_module("actor_network", actor_network);
	    register_module("actor_head", actor_head);
	    register_module("log_std_head", log_std_head);
	    //register_parameter("log_std", log_std_);
	    register_module("critic_network", critic_network);
    }

	void copy_from(const ActorCriticImpl *other)
    {
		//printf("1\n");
	    // Clone the actor network from the other model
		actor_network = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(other->actor_network->clone());
	    //actor_network = *(torch::nn::Sequential*)(other->actor_network->clone().get());
		//printf("1\n");

		// Clone the actor head from the other model
		actor_head = std::dynamic_pointer_cast<torch::nn::LinearImpl>(other->actor_head->clone());

		// Clone the log_std head from the other model
		log_std_head = std::dynamic_pointer_cast<torch::nn::LinearImpl>(other->log_std_head->clone());

	    // Clone the critic network from the other model
		critic_network = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(other->critic_network->clone());
		//printf("1\n");
		//std::cout << other->log_std_ << std::endl;
		//  Clone the critic network from the other model
		lstm = std::dynamic_pointer_cast<torch::nn::LSTMImpl>(other->lstm->clone());
		this->lstm->flatten_parameters();
	    // Copy the log_std_ parameter
	    //log_std_ = other->log_std_.clone();
	    if(!other->is_training())
	    {
		    this->eval();
	    }
	    else
	    {
		    this->train();
	    }
	    //std::cout << log_std_ << std::endl;
    }

	// Actor network parameters
    auto actor_network_parameters()
    {
	    return actor_network->parameters();
    }

	// Actor head parameters
    auto actor_head_parameters()
    {
	    return actor_head->parameters();
    }

	// Log std head parameters
    auto log_std_head_parameters()
    {
	    return log_std_head->parameters();
    }

	// Critic parameters
    auto critic_parameters()
    {
	    return critic_network->parameters();
    }

	// Fast normal without synchronization that torch::normal do
	torch::Tensor fast_normal(torch::Tensor mean, torch::Tensor log_std)
    {
	    // Ensure tensors are on CUDA
	    TORCH_CHECK(mean.is_cuda() && log_std.is_cuda(), "Tensors must be on CUDA");

	    // Generate two uniform random tensors (0,1)
	    // rand_like outputs in range [0,1), so rotate it to avoid log(0) which is inf
	    auto U1 = 1 - torch::rand_like(mean, torch::kCUDA);
	    auto U2 = torch::rand_like(mean, torch::kCUDA);

	    // Box-Muller transform
	    auto R = torch::sqrt(-2.0 * torch::log(U1));
	    auto theta = 2.0 * M_PI * U2;

	    // Two independent normal samples
	    auto Z = R * torch::cos(theta);

		auto std = torch::exp(log_std);

	    // Scale by std and shift by mean
		return Z * std + mean;
    }

	// Forward pass.
    auto normal_angles(torch::Tensor x, torch::Tensor std) -> torch::Tensor
    {
	    if(this->is_training())
	    {
		    torch::Tensor action = x.clone();
		    try
		    {
			    //action = at::normal(x, log_std_.exp().expand_as(x));
			    //if(used_presamples >= count_presampled)
			    //{
				   // printf("presampled normals used off\n");
				   // //presample_normal(normal_presampled.size(0), normal_presampled.size(1));
			    //}
			    action.slice(1, 0, 2) = fast_normal(action.slice(1, 0, 2), std);
			    //action = x + normal_presampled[used_presamples];
			    //used_presamples += 1;
		    }
		    catch(const std::exception &e)
		    {
			    std::cout << "KEK: " << e.what() << std::endl;
			    //std::cout << "Sizes of log_std_:" << log_std_ << std::endl;
			    /*std::cout << std << std::endl;
			    std::cout << std.device() << std::endl;
			    std::cout << std.dtype() << std::endl;
			    std::cout << std.sizes() << std::endl;*/
			    system("PAUSE");
		    }
		    return action;
	    }
	    else
	    {
		    return x;
	    }
    }

    // Initialize network.
    void normal(double mu, double std) 
    {
        torch::NoGradGuard no_grad;

        for (auto& p: this->parameters()) 
        {
            p.normal_(mu,std);
        }         
    }

	// Normal is making synchronization so we need to presample it - https://pytorch.org/docs/stable/generated/torch.normal.html
	//void presample_normal(int count_samples, int count_players)
 //   {
	//    torch::Tensor zero_mean = torch::zeros({count_samples, count_players, 2}, torch::kCUDA);

	//    try
	//    {
	//	    //static double maxi_max = 0;
	//	    normal_presampled = at::normal(zero_mean, log_std_.exp().expand_as(zero_mean));
	//	    /*auto maxee = abs(normal_presampled.max().item<double>());
	//	    if(maxee > 10 && maxee > maxi_max)
	//	    {
	//		    maxi_max = maxee;
	//		    std::cout << "Presample new max: " << maxi_max << std::endl;
	//		    std::cout << log_std_ << std::endl;
	//	    }*/
	//    }
	//    catch(const std::exception &e)
	//    {
	//	    std::cout << "presample_normal error: " << e.what() << std::endl;
	//	    std::cout << log_std_ << std::endl;
	//    }
	//	//std::cout << 
	//    //printf("2\n");

	//    count_presampled = count_samples;
	//    used_presamples = 0;
 //   }

	// Gaussian entropy
    auto entropy_gaussian(torch::Tensor log_std) -> torch::Tensor
    {
	    // Differential entropy of normal distribution. For reference https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal
	    auto gaussian_entropy = 0.5 + 0.5 * log(2 * M_PI) + log_std;
	    //std::cout << gaussian_entropy.sizes() << std::endl;
	    // Sum over the last dimension (angle components)
	    return gaussian_entropy.sum(1); // Shape [...]
    }

	// Bernoulli entropy
    auto entropy_bernoulli(torch::Tensor probs) -> torch::Tensor
    {
	    probs = probs.clamp(EPSILON, 1.f - EPSILON);
	    auto log_probs = torch::log(probs);
	    auto log_1_minus_probs = torch::log(1.0 - probs);
	    return -(probs * log_probs + (1.0 - probs) * log_1_minus_probs);
    }

    // Categorical entropy
    auto entropy_categorical(torch::Tensor probs) -> torch::Tensor
    {
	    probs = probs.clamp(EPSILON);
		auto log_probs = torch::log(probs);
	    return -(probs * log_probs).sum(-1);
    }

    auto entropy(torch::Tensor action) -> torch::Tensor
    {
	    auto log_std = action.slice(1, 7, 9); // Shape [batch_size, 2]
	    auto angle_entropy = entropy_gaussian(log_std);
        
		auto probs = torch::sigmoid(action.slice(1, 5, 6)); // Shape [batch_size, 1]
		auto hook_entropy = entropy_bernoulli(probs);
		probs = torch::sigmoid(action.slice(1, 6, 7)); // Shape [batch_size, 1]
		auto hammer_entropy = entropy_bernoulli(probs);

		probs = torch::softmax(action.slice(1, 2, 5), 1); // Shape [batch_size, 3]
		auto direction_entropy = entropy_categorical(probs);

		hook_entropy = hook_entropy.squeeze(-1); // Convert from [batch_size, 1] to [batch_size]
		hammer_entropy = hammer_entropy.squeeze(-1);

        return angle_entropy + hook_entropy + hammer_entropy + direction_entropy;
    }

	// Extract log probabilities for categorical distribution
    torch::Tensor log_prob_categorical_batch(torch::Tensor logits, torch::Tensor actions)
    {
	    // Clamp logits to prevent extreme values
	    logits = logits.clamp(-100.0, 100.0); // Prevent overflow in softmax
	    // Compute log probabilities for all actions
	    auto log_probs = torch::log_softmax(logits, -1);
	    actions = actions.to(torch::kInt64);

	    // Extract the log probability of the sampled actions
	    return log_probs.gather(-1, actions); // unsqueeze(-1) .squeeze(-1)
    }

	// Extract log probabilities for Bernoulli distribution
    torch::Tensor log_prob_bernoulli_batch(torch::Tensor logits, torch::Tensor actions)
    {
	    // Clamp logits to prevent extreme values
	    logits = logits.clamp(-100.0, 100.0); // Prevent overflow in softmax
	    // Compute probabilities
	    auto probs = torch::sigmoid(logits).clamp(EPSILON, 1.f - EPSILON);

	    // Compute log probabilities
	    return torch::where(
		    actions == 1,
		    torch::log(probs),
		    torch::log(1.0 - probs));
    }

    auto log_prob(torch::Tensor logits, torch::Tensor sampled) -> torch::Tensor
    {
	    auto log_probs = torch::zeros({(int)sampled.size(0), 5}, sampled.device());

		// Apply tanh to first 2 outputs
	    auto mu_ = torch::tanh(logits.slice(1, 0, 2));
	    auto _action = sampled.slice(1, 0, 2);

	    auto log_std = logits.slice(1, 7, 9);

        // Logarithmic probability of taken action, given the current distribution.
		torch::Tensor var = (log_std + log_std).exp();
	    log_probs.slice(1, 0, 2) += -((_action - mu_) * (_action - mu_)) / (2 * var) - log_std - log(sqrt(2 * M_PI));
		log_probs.slice(1, 2, 3) += log_prob_categorical_batch(logits.slice(1, 2, 5), sampled.slice(1, 2, 3));
		log_probs.slice(1, 3, 4) += log_prob_bernoulli_batch(logits.slice(1, 5, 6), sampled.slice(1, 3, 4));
		log_probs.slice(1, 4, 5) += log_prob_bernoulli_batch(logits.slice(1, 6, 7), sampled.slice(1, 4, 5));

        return log_probs.sum(1).reshape({(int)sampled.size(0), 1});
    }
};

TORCH_MODULE(ActorCritic);
