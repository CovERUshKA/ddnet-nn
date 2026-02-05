#pragma once
#include "NNStats.h"

typedef unsigned int uint;

using namespace std;

struct ModelInputInputs
{
	// Indicates on which side bot is located
	// -1 - means left side
	// 1 - means right side
	float side;

	//
	// Local bot
	//

	// Position of the player in the area
	vec2 bot_pos;
	// Indicates whether the bot is out of area
	float bot_is_out_of_area;
	// Velocity of the bot by x and y axis
	vec2 bot_vel;

	// Hammer time (0 - can hammer, 1 - can't hammer)
	float bot_hammer_time;
	// Hook time (0 - hook ended, 1 - hook started)
	float bot_hook_time;

	// Time till bot unfreeze. 0 means unfreezed.
	float bot_freeze_time;

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
	// Indicates whether the enemy is out of area
	float enemy_is_out_of_area;
	// Velocity of the enemy by x and y axis
	vec2 enemy_vel;

	// Hammer time (0 - can hammer, 1 - can't hammer)
	float enemy_hammer_time;
	// Hook time (0 - hook ended, 1 - hook started)
	float enemy_hook_time;

	// Time till enemy unfreeze. 0 means unfreezed.
	float enemy_freeze_time;

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
	// Hook angle according to the enemy tee at the moment
	vec2 enemy_hook_angle;

	//
	// Ball
	//
	// Position of the ball in the area
	vec2 ball_pos;
	// Indicates whether the ball is out of area
	float ball_is_out_of_area;
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
	// Should bot hammer/hold
	bool hammer;
};

struct ModelManager
{
	bool is_training;
	std::string train_folder;
	int count_bots, iReplaysPerBot, batch_size;
	ModelManager(bool is_training, std::string train_folder, size_t batch_size, size_t count_players, uint64_t seed);

	bool LoadModels(std::string folder_path, std::string main_model_name, bool load_previous);
	bool ReloadCachedModels();

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

	void Reward(float reward, bool reset_accumulation, bool done);
	void SaveReplays(bool &is_full);
	void ErasePlayerReplays(int id);

	void Update(double avg_reward, bool cache_model, bool &updated,
		NNStats &stats);

	bool IsOldModel(int id);
	void ReassignOldModels();

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
	// Return entropy coefficient
	double GetEntropyCoefficient();
	// Reset bot memory
	bool ResetBotMemory(int bot_id);
	// Reset memory of all bots
	bool ResetAllBotsMemory();
	void ResetCUDAGraph();
};
