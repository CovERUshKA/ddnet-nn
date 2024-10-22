#pragma once

typedef unsigned int uint;

using namespace std;

struct ModelInputInputs
{
	// Position of the player in the block
	vec2 pos;
	// Position of another bot
	//vec2 pos_2;
	// Velocity of the bot by x and y axis
	vec2 m_vel;
	// Is bot on ground
	float is_grounded;
	// Can bot jump
	float can_jump;
	// Is bot using jump
	float is_jumping;
	
	// 
	// HOOK
	// 
	// Is bot using hook?
	float is_hooking;
	// Is bot hooked something
	float is_grabbed;
	// Is hook retracted
	float is_retracted;
	// Position of hook when it is nearby
	vec2 hook_pos;
	// Direction the hook is going
	vec2 hook_dir;
	// Hook angle according to tee at the moment
	vec2 hook_angle;
	// Old hook angle according to tee
	vec2 hook_old_angle;

	// Path towards the finish containing 30 steps(blocks)
	vec2 path[30]; 
};

struct ModelInputBlocks
{
	// Blocks indexes
	long long blocks[33 * 33];
};

struct ModelOutput
{
	// Angle to point
	vec2 angle;
	/// Which direction should bot go
	/// -1 - left
	/// 0 - stand
	/// 1 - right
	int direction;
	// Should bot hook/hold
	bool hook;
	// Should bot jump
	bool jump;
};

struct ModelManager
{
	int count_bots, iReplaysPerBot;
	ModelManager(size_t batch_size, size_t count_players);

	//ModelOutput Decide(ModelInputInputs &input);
	std::vector<ModelOutput> Decide(
		std::vector<ModelInputInputs> &input,
		std::vector<ModelInputBlocks> &blocks,
		double &time_pre_forward,
		double &time_forward,
		double &time_normal,
		double &time_to_cpu,
		double &time_process_last);
	//std::vector<ModelOutput> Decide(std::vector<ModelInput> &input);

	void Reward(float reward, bool done);
	void SaveReplays();

	void Update(double avg_reward, double &avg_training_loss, double &avg_actor_loss, double &avg_critic_loss);

	void Save(std::string filename);

	bool IsTraining();

	size_t GetCountOfReplays();
	// Return starting learning rate
	double GetLearningRate();
	// Return current learning rate
	double GetCurrentLearningRate();
	// Returns mini batch size
	int64_t GetMiniBatchSize();
	// Returns count of PPO epochs
	int64_t GetCountPPOEpochs();
};
