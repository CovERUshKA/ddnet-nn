#pragma once

#include <random>
#include <base/vmath.h>
#include <engine/map.h>
#include <game/server/player.h>
#include <game/server/entities/character.h>
#include <game/mapitems.h>
#include "ModelManager.h"
#include "AStar.h"
#include <fstream>

class CNeuralNetwork : public IInterface
{
	MACRO_INTERFACE("neuralnetwork", 0)
	
	// Validation phase
	//bool validated;
	//bool validating;
	//int validating_dones;

	bool spawn_probabilities_updated;

	int skip_tick;
	int available_ticks_to_store;
	int count_ticks;
	int update_tick;
	int ticks_collected;
	int last_update_tick;

	std::random_device rd;
	std::mt19937 gen;
	std::discrete_distribution<> spawn_probabilities_distribution;

	std::vector<vec2> vSpawnPoints;
	std::vector<std::pair<int, int>> vFinishPoses;
	std::vector<std::vector<int>> pathfinding_grid;
	std::vector<unsigned char> map_game_grid;

	std::vector<float> vSpawnCumulativeReward;
	std::vector<int> vSpawnLives;
	std::vector<float> vSpawnProbabilities;

	int count_bots;
	std::vector<CPlayer *> vBots;
	std::vector<std::vector<std::pair<int, int>>> vBotsPath;
	std::vector<vec2> vBotLastPos;
	std::vector<float> vBotLastVel;
	std::vector<int> vBotsSpawnPos;
	//std::vector<int> vBotsValidateSpawnPoint;
	std::vector<vec2> vBotsLastCheckPoint;
	std::vector<float> vBotsCumulativeRewards;
	std::vector<ModelInputInputs> vInputInputs;
	//std::vector<ModelInputBlocks> vInputBlocks;
	std::vector<ModelOutput> vOutputs;
	//std::vector<bool> vIsPreviouslyHooked;
	//std::vector<vec2> vPrevHookPos;

	// First is distance, second is tick
	std::vector<std::pair<int, int>> vBotBestDistance;

	AStar *astar;

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

	CPlayer *AddBot(const char *Name);
}; // namespace NeuralNetwork
