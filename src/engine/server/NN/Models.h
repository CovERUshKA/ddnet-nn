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
	int64_t n_in, n_out, used_presamples;

    // Actor.
	//torch::nn::Linear a_lin1_, a_lin2_, /*a_lin3_,*/ a_lin4_;
    torch::nn::Sequential actor_network;
    torch::Tensor mu_;
    torch::Tensor log_std_, normal_presampled;

    // Critic.
    //torch::nn::Linear c_lin1_, c_lin2_, /*c_lin3_,*/ c_lin4_, c_val_;
    torch::nn::Sequential critic_network;

    ActorCriticImpl(int64_t n_in, int64_t n_out, double std) :
	    n_in(n_in), n_out(n_out),
		// Actor.
       //   a_lin1_(torch::nn::Linear(n_in, 16)),
       //   a_lin2_(torch::nn::Linear(16, 32)),
	      ////a_lin3_(torch::nn::Linear(16, 16)),
       //   a_lin4_(torch::nn::Linear(32, n_out)),
	    actor_network(torch::nn::Sequential(
		    torch::nn::Linear(n_in, 2048),
		    torch::nn::ReLU(),
			torch::nn::Linear(2048, 1024),
		    torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    torch::nn::Linear(1024, 512),
		    torch::nn::ReLU(),
		    // torch::nn::Dropout(0.2),
		    torch::nn::Linear(512, 256),
			torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    torch::nn::Linear(256, 128),
		    torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    //torch::nn::Linear(1024, 1024),
			//torch::nn::ReLU(),
		    // torch::nn::Dropout(0.2),
		    //torch::nn::Linear(1024, 1024),
			//torch::nn::ReLU(),
		    // torch::nn::Dropout(0.2),
		    //torch::nn::Linear(1024, 1024),
			//torch::nn::ReLU(),
		    // torch::nn::Dropout(0.2),
		    //torch::nn::Linear(256, 128),
		    //torch::nn::ReLU(),
		    torch::nn::Linear(128, n_out)/*,
			torch::nn::Tanh()*/)),
          mu_(torch::full(n_out, 0.)),
          log_std_(torch::full(n_out, std)),
	    critic_network(torch::nn::Sequential(
			torch::nn::Linear(n_in, 2048),
		    torch::nn::ReLU(),
			//torch::nn::Dropout(0.2),
		    torch::nn::Linear(2048, 1024),
		    torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    torch::nn::Linear(1024, 512),
		    torch::nn::ReLU(),
			//torch::nn::Dropout(0.2),
		    torch::nn::Linear(512, 256),
			torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    torch::nn::Linear(256, 128),
		    torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    //torch::nn::Linear(1024, 1024),
			//torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    //torch::nn::Linear(1024, 1024),
			//torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    //torch::nn::Linear(1024, 1024),
		    //torch::nn::ReLU(),
		    // torch::nn::Dropout(0.2),
		    //torch::nn::Linear(128, 64),
		    //torch::nn::ReLU(),
		    //torch::nn::Dropout(0.2),
		    //torch::nn::Linear(64, n_out),
			//torch::nn::ReLU(),
		    torch::nn::Linear(128, 1)
        ))
          
          // Critic
	   // c_lin1_(torch::nn::Linear(n_in, 16)),
	   // c_lin2_(torch::nn::Linear(16, 32)),
	   //// c_lin3_(torch::nn::Linear(32, 16)),
	   // c_lin4_(torch::nn::Linear(32, n_out)),
    //      c_val_(torch::nn::Linear(n_out, 1)) 
    {
	    //register_module("conv_layers", conv_layers);
	    //register_module("scalar_fc_layers", scalar_fc_layers);
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
    auto actor_forward(torch::Tensor x) -> torch::Tensor
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
		    // std::lock_guard<std::mutex> guard(g_mutex);
		    //torch::NoGradGuard no_grad;
		    //printf("1\n");
		    //auto ten = x.to(torch::kCPU);

		    //auto decide_time = std::chrono::high_resolution_clock::now();
		    //log_std = log_std_cpu.to(torch::kCPU);
		    //printf("1\n");
		    //x = x.to(torch::kCPU, true);

		    //auto std = log_std_.to(torch::kCPU, true); // .to(torch::kCPU)
		    //at::cuda::stream_synchronize(at::cuda::getCurrentCUDAStream());
		    //auto std = log_std_.exp().expand_as(x);
		    //printf("2\n");
		    //std::cout << std << std::endl;

		    //printf("3\n");
		    //at::cuda::stream_synchronize(stream);
		    //  Create a mask where tensor values are less than 0
		    //auto mask = std < 0;

		    // Print the mask to show which elements are negative
		    //std::cout << "Mask (True indicates negative values): " << mask << std::endl;

		    // Extract the negative elements using the mask
		   // auto negative_elements = std.index({mask});

		    // Print the negative elements
		    //std::cout << "Negative Elements: " << negative_elements.numel() << std::endl;
		    //if(negative_elements.numel())
		    //{
			    //std::cout << "Negative Elements: " << negative_elements << std::endl;
		    //}
		    //auto std_cpu = std.contiguous().to(torch::kCPU); // .to(torch::kCPU)
		    /*std::cout << x.sizes() << std::endl;
		    std::cout << std.sizes() << std::endl;
		    std::cout << x.device() << std::endl;
		    std::cout << std.device() << std::endl;
		    std::cout << x.scalar_type() << std::endl;
		    std::cout << std.scalar_type() << std::endl;
		    printf("33.0\n");*/
		    //auto end = std::chrono::high_resolution_clock::now();
		    //auto ten = x.to(torch::kCPU);
		    //printf("33.1\n");
		    //std::chrono::duration<double> elapsed = end - decide_time;
		    //std::cout << "Elapsed std: " << elapsed.count() << " seconds" << std::endl;
		    //std::cout << std_cpu.sizes() << std::endl;
		    //torch::Tensor action = at::normal(x.to(torch::kCPU), std);
			//decide_time = std::chrono::high_resolution_clock::now();
			
		    //std::cout << "Contains: " << std[0][0].item<float>() << std::endl;

		    torch::Tensor action;
		    try
		    {
			    action = x + normal_presampled[used_presamples]; // at::normal(x, std);
			    used_presamples += 1;
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
		    //std::cout << "Contains after: " << std[0][0].item<float>() << std::endl;

		    //printf("4\n");

			//end = std::chrono::high_resolution_clock::now();
		    //elapsed = end - decide_time;
			//std::cout << "Elapsed normal: " << elapsed.count() << " seconds" << std::endl;
		    //printf("2\n");
		    // action = torch::tanh(action); // Squash the sampled action to be within the range [-1, 1]
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

	/*void log_to_cpu()
    {
		log_std_cpu = log_std_.to(torch::kCPU, true);
	    at::cuda::stream_synchronize(at::cuda::getCurrentCUDAStream());
    }*/

	void presample_normal(int count_samples, int count_players)
    {
		static torch::Tensor zero_mean = torch::zeros({count_samples, count_players, n_out}, torch::kCUDA);
	    normal_presampled = at::normal(zero_mean, log_std_.exp().expand_as(zero_mean));
	    used_presamples = 0;
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
