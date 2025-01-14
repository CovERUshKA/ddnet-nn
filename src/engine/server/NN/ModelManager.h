#pragma once

typedef unsigned int uint;

using namespace std;

struct ModelInputInputs
{
	//
	// Local bot
	//

	// Position of the player in the area
	vec2 bot_pos;
	// Velocity of the bot by x and y axis
	vec2 bot_vel;

	// HOOK

	// Is bot using hook?
	float bot_is_hooking;
	// Is bot hooked something
	float bot_is_grabbed;
	// Is hook retracted
	float bot_is_retracted;
	// Position of hook
	vec2 bot_hook_pos;
	// Direction the hook is going
	vec2 bot_hook_dir;
	// Hook angle according to tee at the moment
	vec2 bot_hook_angle;

	//
	// Enemy
	//
	
	// Position of the enemy in the area
	vec2 enemy_pos;
	// Velocity of the enemy by x and y axis
	vec2 enemy_vel;

	// HOOK

	// Is enemy using hook?
	float enemy_is_hooking;
	// Is enemy hooked something
	float enemy_is_grabbed;
	// Is hook retracted
	float enemy_is_retracted;
	// Position of hook
	vec2 enemy_hook_pos;
	// Direction the hook is going
	vec2 enemy_hook_dir;

	//
	// Ball
	//
	// Position of the ball in the area
	vec2 ball_pos;
	// Velocity of the enemy by x and y axis
	vec2 ball_vel;
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
};
struct ModelManager
{
	int count_bots, iReplaysPerBot, batch_size;
	ModelManager(std::vector<unsigned char> &map_game_grid, int map_width, int map_height, size_t batch_size, size_t count_players, uint64_t seed);

	//ModelOutput Decide(ModelInputInputs &input);
	std::vector<ModelOutput> Decide(
		std::vector<ModelInputInputs> &input,
		double &time_pre_forward,
		double &time_forward,
		double &time_normal,
		double &time_to_cpu,
		double &time_process_last,
		bool validating = false);
	//std::vector<ModelOutput> Decide(std::vector<ModelInput> &input);

	void Reward(float reward, bool done);
	void SaveReplays(bool &is_full);
	void ErasePlayerReplays(int id);

	void Update(double avg_reward, int dies, bool spawn_probabilities_updated, bool &updated, double &avg_training_loss, double &avg_actor_loss, double &avg_critic_loss);

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
	size_t GetCountEpisodes();
};
