#pragma once

#include <random>
#include <base/vmath.h>
#include <engine/map.h>
#include <game/server/player.h>
#include <game/server/entities/character.h>
#include <game/mapitems.h>
#include "ModelManager.h"
#include <fstream>
#include <deque>

class CNeuralNetwork : public IInterface
{
	MACRO_INTERFACE("neuralnetwork", 0)

	bool is_training;
	int skip_tick;
	int available_ticks_to_store;
	int count_ticks;
	int update_tick;
	int ticks_collected;
	int last_update_tick;
	int cache_model_gap;

	//std::random_device rd;
	//std::mt19937 gen;
	//std::discrete_distribution<> spawn_probabilities_distribution;

	int count_bots;
	int count_teams;
	int count_player_bots;
	std::vector<CPlayer *> vBots;
	//std::vector<vec2> vBotLastPos;
	std::vector<vec2> vBallLastPos;
	//std::vector<float> vBotLastVel;
	//std::vector<int> vBotsSpawnPos;
	//std::vector<float> vBotsCumulativeRewardBetweenSkipTick;
	//std::vector<float> vBotsCumulativeRewards;
	std::vector<ModelInputInputs> vInputInputs;
	std::vector<ModelOutput> vOutputs;

	// First is distance, second is tick
	//std::vector<std::pair<int, int>> vBotBestDistance;

	ModelManager* model_manager;

	// train directory name
	std::string dir_name;

	std::ofstream logger;

	std::chrono::high_resolution_clock::time_point decide_time;
	float cumulative_time_to_decide;
	float cumulative_time_to_tick;
	float cumulative_time_rest;

	// Decide times
	double cumulative_time_pre_forward;
	double cumulative_time_forward;
	double cumulative_time_normal;
	double cumulative_time_to_cpu;
	double cumulative_time_process_last;

	CMapItemLayerTilemap *gamelayer;
	CTile *pTiles;

	// IF set to true - respawns all
	bool respawn_all;

	// Calculate ticks per second
	int64_t ticks_timer;
	int start_ticks;
	int ticks_per_second;

public:
	void OnInit();

	void PreTick();
	void PostTick(float time_to_tick);

	void PreOnClientPredictedEarlyInput();
	void PreOnClientPredictedInput();

	bool IsTraining();

	void RespawnTeam(int team);
	void StartFight(CPlayer* player, bool right_side);

	bool IsSwitchEnabled(int Number, int Team);
	void ChangeSwitchState(int Number, int Team, bool state);

	CPlayer *AddBot(std::string name, vec2 spawn_pos);
}; // namespace NeuralNetwork
