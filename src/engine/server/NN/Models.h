#pragma once

#include <torch/torch.h>
#include <math.h>

// Network model for Proximal Policy Optimization on Incy Wincy.
struct ActorCriticImpl : public torch::nn::Module 
{
    // Actor.
	//torch::nn::Linear a_lin1_, a_lin2_, /*a_lin3_,*/ a_lin4_;
    torch::nn::Sequential actor_network;
    torch::Tensor mu_;
    torch::Tensor log_std_;

    // Critic.
    //torch::nn::Linear c_lin1_, c_lin2_, /*c_lin3_,*/ c_lin4_, c_val_;
    torch::nn::Sequential critic_network;

    ActorCriticImpl(int64_t n_in, int64_t n_out, double std)
        : // Actor.
       //   a_lin1_(torch::nn::Linear(n_in, 16)),
       //   a_lin2_(torch::nn::Linear(16, 32)),
	      ////a_lin3_(torch::nn::Linear(16, 16)),
       //   a_lin4_(torch::nn::Linear(32, n_out)),
	    actor_network(torch::nn::Sequential(
		    torch::nn::Linear(n_in, 2048),
		    torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    torch::nn::Linear(2048, 1024),
		    torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    //torch::nn::Linear(1024, 512),
		    //torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    torch::nn::Linear(1024, n_out),
			torch::nn::Tanh())),
          mu_(torch::full(n_out, 0.)),
          log_std_(torch::full(n_out, std)),
	    critic_network(torch::nn::Sequential(
		    torch::nn::Linear(n_in, 1024),
		    torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    torch::nn::Linear(1024, 512),
		    torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    //torch::nn::Linear(512, 256),
		    //torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    torch::nn::Linear(512, n_out),
		    torch::nn::ReLU(),
		    torch::nn::Linear(n_out, 1)
        ))
          
          // Critic
	   // c_lin1_(torch::nn::Linear(n_in, 16)),
	   // c_lin2_(torch::nn::Linear(16, 32)),
	   //// c_lin3_(torch::nn::Linear(32, 16)),
	   // c_lin4_(torch::nn::Linear(32, n_out)),
    //      c_val_(torch::nn::Linear(n_out, 1)) 
    {
	    register_module("actor_network", actor_network);
        // Register the modules.
     //   register_module("a_lin1", a_lin1_);
     //   register_module("a_lin2", a_lin2_);
     //   //register_module("a_lin3", a_lin3_);
	    //register_module("a_lin4", a_lin4_);
        register_parameter("log_std", log_std_);
	    register_module("critic_network", critic_network);

     //   register_module("c_lin1", c_lin1_);
     //   register_module("c_lin2", c_lin2_);
     //   //register_module("c_lin3", c_lin3_);
	    //register_module("c_lin4", c_lin4_);
     //   register_module("c_val", c_val_);
    }

    // Forward pass.
    auto forward(torch::Tensor x) -> std::tuple<torch::Tensor, torch::Tensor> 
    {
	    //torch::NoGradGuard no_grad;
        // Actor.
	    //printf("1\n");
        //mu_ = torch::relu(a_lin1_->forward(x));
	    //printf("2\n");
        //mu_ = torch::relu(a_lin2_->forward(mu_));
	    //mu_ = torch::relu(a_lin3_->forward(mu_));
	    //mu_ = torch::tanh(a_lin4_->forward(mu_));
	    mu_ = actor_network->forward(x);
        // Critic.
        //torch::Tensor val = torch::relu(c_lin1_->forward(x));
        //val = torch::relu(c_lin2_->forward(val));
	    ////val = torch::relu(c_lin3_->forward(val));
	    //val = torch::relu(c_lin4_->forward(val));
        //val = c_val_->forward(val);
	    torch::Tensor val = critic_network->forward(x);

        if (this->is_training()) 
        {
		    //std::lock_guard<std::mutex> guard(g_mutex);
            torch::NoGradGuard no_grad;
		    //printf("1\n");
            torch::Tensor action = at::normal(mu_, log_std_.exp().expand_as(mu_));
		    //printf("2\n");
	        //action = torch::tanh(action); // Squash the sampled action to be within the range [-1, 1]
            return std::make_tuple(action, val);  
        }
        else
        {
            return std::make_tuple(mu_, val);
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
