#pragma once

#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <math.h>
#include <ctime>
#include <c10/cuda/CUDACachingAllocator.h>

// Network model for Proximal Policy Optimization on Incy Wincy.
struct ActorCriticImpl : public torch::nn::Module 
{
	int64_t n_in, n_out;

    // Actor.
    torch::nn::Sequential actor_network;
    torch::Tensor mu_;
    torch::Tensor log_std_;

    // Critic.
    torch::nn::Sequential critic_network;

	ActorCriticImpl()
	{

	}

    bool Initialize(int64_t n_in, int64_t n_out, double std)
    {
	    n_in = n_in;
	    n_out = n_out;
	    actor_network = torch::nn::Sequential(
		    torch::nn::Linear(n_in, 2048),
		    torch::nn::ReLU(),
		    torch::nn::Linear(2048, 1024),
		    torch::nn::ReLU(),
		    torch::nn::Linear(1024, 512),
		    torch::nn::ReLU(),
		    torch::nn::Linear(512, 256),
		    torch::nn::ReLU(),
		    torch::nn::Linear(256, 128),
		    torch::nn::ReLU(),
		    torch::nn::Linear(128, n_out)
		    );
		mu_ = torch::full(n_out, 0.);
	    log_std_ = torch::full(n_out, std);
		critic_network = torch::nn::Sequential(
			torch::nn::Linear(n_in, 2048),
			torch::nn::ReLU(),
			torch::nn::Linear(2048, 1024),
			torch::nn::ReLU(),
			torch::nn::Linear(1024, 512),
			torch::nn::ReLU(),
			torch::nn::Linear(512, 256),
			torch::nn::ReLU(),
			torch::nn::Linear(256, 128),
			torch::nn::ReLU(),
			torch::nn::Linear(128, 1));

	    register_module("actor_network", actor_network);
        register_parameter("log_std", log_std_);
	    register_module("critic_network", critic_network);
	

    }

    // Forward pass.
    auto actor_forward(torch::Tensor x) -> torch::Tensor
    {
	    //torch::NoGradGuard no_grad;
        // Actor.
	    try
	    {
		    mu_ = actor_network->forward(x);
	    }
	    catch(const std::exception &e)
	    {
		    std::cout << "actor_network->forward crashed with reason: " << e.what() << std::endl;
		    exit(1);
	    }

	    return mu_;
    }

    // Forward pass.
    auto critic_forward(torch::Tensor x) -> torch::Tensor
    {
	    // Critic.

	    // torch::Tensor val = torch::relu(c_lin1_->forward(x));
	    // val = torch::relu(c_lin2_->forward(val));
	    ////val = torch::relu(c_lin3_->forward(val));
	    // val = torch::relu(c_lin4_->forward(val));
	    // val = c_val_->forward(val);
		torch::Tensor val = critic_network->forward(x);
	    return val;
    }

	// Copy constructor
    ActorCriticImpl(const ActorCriticImpl *other)
    {
	    actor_network = *(torch::nn::Sequential*)(other->actor_network->clone().get());
	    log_std_ = other->log_std_.clone();
	    critic_network = *(torch::nn::Sequential*)(other->critic_network->clone().get());
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

	    // Copy the log_std_ parameter
	    log_std_ = other->log_std_.clone();
	    if(!other->is_training())
	    {
		    this->eval();
	    }
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

	// Forward pass.
    auto normal_actor(torch::Tensor x) -> torch::Tensor
    {
	    if(this->is_training())
	    {
		    torch::Tensor action;
		    try
		    {
			    //at::manual_seed()
			    action = at::normal(x, log_std_.exp().expand_as(x));
		    }
		    catch(const std::exception &e)
		    {
			    std::cout << "KEK: " << e.what() << std::endl;
			    std::cout << log_std_ << std::endl;
			    /*std::cout << std << std::endl;
			    std::cout << std.device() << std::endl;
			    std::cout << std.dtype() << std::endl;
			    std::cout << std.sizes() << std::endl;*/
			    exit(1);
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

    auto entropy() -> torch::Tensor
    {
        // Differential entropy of normal distribution. For reference https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal
        return 0.5 + 0.5*log(2*M_PI) + log_std_;
    }

    auto log_prob(torch::Tensor action) -> torch::Tensor
    {
        // Logarithmic probability of taken action, given the current distribution.
	    torch::Tensor var = (log_std_ + log_std_).exp();

        return -((action - mu_)*(action - mu_)) / (2 * var) - log_std_ - log(sqrt(2 * M_PI));
    }
};

TORCH_MODULE(ActorCritic);
