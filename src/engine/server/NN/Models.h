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
	int64_t n_in, n_out;

    // Actor.
    torch::nn::Sequential actor_network;
    //torch::Tensor mu_;
    torch::Tensor log_std_;

    // Critic.
    torch::nn::Sequential critic_network;

	ActorCriticImpl()
	{

	}

    bool Initialize(int64_t n_in, int64_t n_out, int64_t h_start, double std)
    {
	    this->n_in = n_in;
	    this->n_out = n_out;
	    actor_network = torch::nn::Sequential(
		    torch::nn::Linear(n_in, h_start),
		    torch::nn::ReLU(),
		    torch::nn::Linear(h_start, h_start/2),
			torch::nn::ReLU(),
		    torch::nn::Linear(h_start / 2, h_start/4),
		    torch::nn::ReLU(),
		    torch::nn::Linear(h_start / 4, h_start/8),
		    torch::nn::ReLU(),
		    torch::nn::Linear(h_start / 8, h_start / 16),
		    torch::nn::ReLU(),
		    torch::nn::Linear(h_start / 16, n_out)
		    //torch::nn::Tanh()
		    );
		//mu_ = torch::full(n_out, 0.);
	    log_std_ = torch::full(2, std::log(std));
		critic_network = torch::nn::Sequential(
		    torch::nn::Linear(n_in, h_start),
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

	    //printf("Created from 0\n");

	    register_module("actor_network", actor_network);
        register_parameter("log_std", log_std_);
	    register_module("critic_network", critic_network);
	
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
		    action = actor_network->forward(x);
	    }
	    catch(const std::exception &e)
	    {
		    std::cout << "actor_network->forward crashed with reason: " << e.what() << std::endl;
		    system("PAUSE");
	    }

	    return action;
    }

    // Forward pass.
    auto critic_forward(torch::Tensor x) -> torch::Tensor
    {
	    // Critic.
		torch::Tensor val = critic_network->forward(x);
	    return val;
    }

	// Copy constructor
    ActorCriticImpl(const ActorCriticImpl *other)
    {
	    // Clone the actor network from the other model
	    actor_network = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(other->actor_network->clone());

	    // Clone the critic network from the other model
	    critic_network = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(other->critic_network->clone());

	    // Copy the log_std_ parameter
	    log_std_ = other->log_std_.detach().clone();
	    if(!other->is_training())
	    {
		    this->eval();
	    }
	    else
	    {
		    this->train();
	    }
	    //printf("Copied\n");
	    register_module("actor_network", actor_network);
	    register_parameter("log_std", log_std_);
	    register_module("critic_network", critic_network);
    }

	void copy_from(const ActorCriticImpl *other)
    {
		//printf("1\n");
	    // Clone the actor network from the other model
		actor_network = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(other->actor_network->clone());
	    //actor_network = *(torch::nn::Sequential*)(other->actor_network->clone().get());
		//printf("1\n");

	    // Clone the critic network from the other model
		critic_network = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(other->critic_network->clone());
		//printf("1\n");
		//std::cout << other->log_std_ << std::endl;

	    // Copy the log_std_ parameter
	    log_std_ = other->log_std_.clone();
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

	// Forward pass.
    auto actor_parameters()
    {
	    return actor_network->parameters();
    }

	// Forward pass.
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
    auto normal_angles(torch::Tensor x) -> torch::Tensor
    {
	    if(this->is_training())
	    {
		    torch::Tensor action = x.clone();
		    try
		    {
			    //action = at::normal(x, log_std_.exp().expand_as(x));
			    action.slice(1, 0, 2) = fast_normal(action.slice(1, 0, 2), log_std_);
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

	// Gaussian entropy
    auto entropy_gaussian() -> torch::Tensor
    {
	    // Differential entropy of normal distribution. For reference https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal
	    auto gaussian_entropy = 0.5 + 0.5 * log(2 * M_PI) + log_std_;
	    //std::cout << gaussian_entropy.sizes() << std::endl;
	    // Sum over the last dimension (angle components)
	    return gaussian_entropy.sum(); // Shape [...]
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
	    auto angle_entropy = entropy_gaussian();
        
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

        // Logarithmic probability of taken action, given the current distribution.
	    torch::Tensor var = (log_std_ + log_std_).exp();
	    log_probs.slice(1, 0, 2) += -((_action - mu_) * (_action - mu_)) / (2 * var) - log_std_ - log(sqrt(2 * M_PI));
		log_probs.slice(1, 2, 3) += log_prob_categorical_batch(logits.slice(1, 2, 5), sampled.slice(1, 2, 3));
		log_probs.slice(1, 3, 4) += log_prob_bernoulli_batch(logits.slice(1, 5, 6), sampled.slice(1, 3, 4));
		log_probs.slice(1, 4, 5) += log_prob_bernoulli_batch(logits.slice(1, 6, 7), sampled.slice(1, 4, 5));

        return log_probs.sum(1).reshape({(int)sampled.size(0), 1});
    }
};

TORCH_MODULE(ActorCritic);
