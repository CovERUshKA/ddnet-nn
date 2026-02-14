#pragma once
#include <optional>
#include <fstream>
#include <iostream>


//struct NNStats
//{
//	// Losses
//	double avg_training_loss;
//	double avg_actor_loss;
//	double avg_critic_loss;
//	// Grads
//	double avg_actor_grad_norm;
//	double avg_critic_grad_norm;
//	double avg_lstm_grad_norm;
//	double avg_actor_head_grad_norm;
//	double avg_log_std_head_grad_norm;
//	// Weights
//	double avg_actor_weight_norm;
//	double avg_critic_weight_norm;
//	// Activation
//	double avg_actor_activation_mean;
//	double avg_actor_activation_std;
//
//	double critic_mean_absolute_error;
//	double critic_correlation_coefficient;
//	// Average entropies
//	double avg_entropy;
//	double avg_angle_entropy;
//	double avg_hook_entropy;
//	double avg_hammer_entropy;
//	double avg_direction_entropy;
//	// Minimal entropies
//	double min_entropy;
//	double min_angle_entropy;
//	double min_hook_entropy;
//	double min_hammer_entropy;
//	double min_direction_entropy;
//	// Maximal entropies
//	double max_entropy;
//	double max_angle_entropy;
//	double max_hook_entropy;
//	double max_hammer_entropy;
//	double max_direction_entropy;
//
//	// Policy Probability Ratio
//	double mean_ratio;
//	double std_ratio;
//	double min_ratio;
//	double max_ratio;
//
//	// Approximate KL Divergence
//	double approx_kl;
//};

struct NNStats
{
private:
	std::ofstream stats_logger;

	// Header order matters for CSV
	std::vector<std::string> headers;

	// Values (None = empty)
	std::map<std::string, std::optional<double>> current_values;

	bool file_opened = false;

public:

	// Register headers
	void add_headers(const std::vector<std::string> &new_headers)
	{
		headers = new_headers;
		for(const auto &h : headers)
			current_values[h] = std::nullopt;
	}

	bool open_file(const std::string &filename, bool dump_headers = true, std::ios_base::openmode mode = std::ios::out)
	{
		stats_logger.open(filename, mode);
		if(!stats_logger.is_open())
			return false;

		file_opened = true;

		if(dump_headers)
		{
			for(size_t i = 0; i < headers.size(); ++i)
			{
				stats_logger << headers[i];
				if(i + 1 < headers.size())
					stats_logger << ",";
			}
			stats_logger << "\n";
			// Print it to file immediately
			stats_logger.flush();
		}

		return true;
	}

	// Set value by header name
	void set(const std::string &key, double value)
	{
		auto it = current_values.find(key);
		if(it != current_values.end())
			it->second = value;
		else
		{
			std::cout << "No such key in NNStats : " << key << std::endl;
		}
	}

	// Get value by header name
	double get(const std::string &key)
	{
		return current_values[key].value_or(0.0);
	}

	// Dump one CSV row and clear values
	void dump()
	{
		if(!file_opened)
			return;

		for(size_t i = 0; i < headers.size(); ++i)
		{
			const auto &key = headers[i];
			const auto &val = current_values[key];

			if(val.has_value())
				stats_logger << *val;

			if(i + 1 < headers.size())
				stats_logger << ",";
		}

		stats_logger << "\n";
		// Print it to file immediately
		stats_logger.flush();

		// Clear values (make them None again)
		for(auto &[_, v] : current_values)
			v = std::nullopt;
	}
};
