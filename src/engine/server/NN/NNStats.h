#pragma once

struct NNStats
{
	// Losses
	double avg_training_loss;
	double avg_actor_loss;
	double avg_critic_loss;
	// Grads
	double avg_actor_grad_norm;
	double avg_critic_grad_norm;
	// Weights
	double avg_actor_weight_norm;
	double avg_critic_weight_norm;
	// Activation
	double avg_actor_activation_mean;
	double avg_actor_activation_std;

	double critic_mean_absolute_error;
	double critic_correlation_coefficient;
	// Average entropies
	double avg_entropy;
	double avg_angle_entropy;
	double avg_hook_entropy;
	double avg_hammer_entropy;
	double avg_direction_entropy;
	// Minimal entropies
	double min_entropy;
	double min_angle_entropy;
	double min_hook_entropy;
	double min_hammer_entropy;
	double min_direction_entropy;
	// Maximal entropies
	double max_entropy;
	double max_angle_entropy;
	double max_hook_entropy;
	double max_hammer_entropy;
	double max_direction_entropy;
};