#include "NeuralNetwork.h"

#include <iostream>
#include <base/logger.h>
#include <base/math.h>
#include <base/vmath.h>

#include <engine/server.h>
#include <engine/console.h>
#include <engine/engine.h>
#include <engine/server/server.h>
#include <engine/shared/config.h>
#include <engine/shared/datafile.h>
#include <engine/shared/json.h>
#include <engine/shared/linereader.h>
#include <engine/shared/memheap.h>
#include <engine/storage.h>
#include <engine/kernel.h>
#include <game/server/gamecontext.h>
#include <game/server/gamecontroller.h>
#include <engine/map.h>
#include "engine/server/server.h"

#include <numeric>
//#include <iostream>

using namespace std;

CServer *m_pServer;
IStorage *m_pStorage;
IConsole *m_pConsole;
CGameContext *m_pGameContext;

CPlayer *CNeuralNetwork::AddBot(const char *Name)
{
	for(int ClientID = MAX_CLIENTS - 1; ClientID >= 0; ClientID--)
	{
		if(m_pServer->m_aClients[ClientID].m_State == CServer::CClient::STATE_EMPTY)
		{
			// m_aClients[ClientID].m_aName = "Bot";
			memcpy(m_pServer->m_aClients[ClientID].m_aName, Name, strlen(Name) + 1);

			m_pServer->m_aClients[ClientID].m_State = CServer::CClient::STATE_INGAME;
			m_pServer->m_aClients[ClientID].m_SnapRate = CServer::CClient::SNAPRATE_FULL;
			m_pServer->m_aClients[ClientID].m_DDNetVersion = VERSION_DDRACE;
			m_pServer->m_aClients[ClientID].m_pRconCmdToSend = 0;

			m_pGameContext->OnClientConnected(ClientID, nullptr);
			m_pGameContext->OnClientEnter(ClientID);

			CPlayer *pPlayer = m_pGameContext->m_apPlayers[ClientID];
			pPlayer->SetAfk(false);
			pPlayer->TryRespawn();

			/*char aFilename[IO_MAX_PATH_LENGTH];
			str_format(aFilename, sizeof(aFilename), "demos/%s_%d_%d_%llu.demo", m_aCurrentMap, m_NetServer.Address().port, ClientID, time_get());
			m_aDemoRecorder[ClientID].Start(Storage(), Console(), aFilename, GameServer()->NetVersion(), m_aCurrentMap, &m_aCurrentMapSha256[MAP_TYPE_SIX], m_aCurrentMapCrc[MAP_TYPE_SIX], "server", m_aCurrentMapSize[MAP_TYPE_SIX], m_apCurrentMapData[MAP_TYPE_SIX]);*/

			return pPlayer;
		}
	}

	return nullptr;
}

void CNeuralNetwork::OnInit()
{
	m_pServer = (CServer*)Kernel()->RequestInterface<IServer>();
	m_pStorage = Kernel()->RequestInterface<IStorage>();
	m_pConsole = Kernel()->RequestInterface<IConsole>();
	m_pGameContext = (CGameContext *)Kernel()->RequestInterface<IGameServer>();

	//validated = false;
	//validating = false;
	//validating_dones = 0;

	skip_tick = 3;
	count_bots = 128;
	available_ticks_to_store = 512000;
	count_ticks = available_ticks_to_store / count_bots;
	update_tick = count_ticks * skip_tick;
	ticks_collected = 0;

	gen = mt19937(rd());

	const CMapItemLayerTilemap *pTileMap = m_pGameContext->Layers()->GameLayer();
	const CTile *pTiles = static_cast<CTile *>(Kernel()->RequestInterface<IMap>()->GetData(pTileMap->m_Data));

	pathfinding_grid.resize(pTileMap->m_Height);

	for(int y = 0; y < pTileMap->m_Height; y++)
	{
		pathfinding_grid[y].resize(pTileMap->m_Width);
		for(int x = 0; x < pTileMap->m_Width; x++)
		{
			const int Index = y * pTileMap->m_Width + x;
			const int GameIndex = pTiles[Index].m_Index - ENTITY_OFFSET;

			switch(pTiles[Index].m_Index)
			{
			case 1:
			case 2:
			case 3:
			case 9:
				pathfinding_grid[y][x] = 1;
				break;
			default:
				break;
			}

			if(pTiles[Index].m_Index == 34)
			{
				vFinishPoses.push_back({y, x});
			}

			// Game layer
			{
				const vec2 Pos(x * 32.0f + 16.0f, y * 32.0f + 16.0f);

				if(pTiles[Index].m_Index >= ENTITY_OFFSET && pTiles[Index].m_Index - ENTITY_OFFSET == ENTITY_SPAWN)
				{
					vSpawnPoints.push_back(Pos);
				}
			}
		}
	}

	//printf("Creating pathfinder...\n");
	dbg_msg("neuralnetwork", "Creating pathfinder...");
	astar = new AStar(pathfinding_grid, vFinishPoses);
	printf("Pathfinder created.\n");
	dbg_msg("neuralnetwork", "Pathfinder created.");


	//printf("Creating train directory with folders...\n");
	dbg_msg("neuralnetwork", "Creating train directory with folders...");

	dir_name = to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());

	if(fs_makedir("train") != 0)
	{
		dbg_msg("neuralnetwork", "Can't make train directory");
		exit(1);
	}

	if(fs_makedir((string("train\\") + dir_name).c_str()) != 0)
	{
		dbg_msg("neuralnetwork", "Can't make dir for this learning directory");
		exit(1);
	}

	if(fs_makedir(string("train\\" + dir_name + "\\models").c_str()) != 0)
	{
		dbg_msg("neuralnetwork", "Can't make models directory");
		exit(1);
	}

	if(fs_makedir(string("train\\" + dir_name + "\\demos").c_str()) != 0)
	{
		dbg_msg("neuralnetwork", "Can't make demos directory");
		exit(1);
	}

	//printf("Train directory with folders created.\n");
	dbg_msg("neuralnetwork", "Train directory with folders created.");

	// std::pair<int, int> start = {4, 4};
	// std::pair<int, int> goal = {4, 62};

	// std::vector<std::pair<int, int>> path = a_star(start, goal, pathfinding_grid);

	// if(!path.empty())
	//{
	//	std::cout << "Path found:\n";
	//	for(const auto &step : path)
	//	{
	//		std::cout << "(" << step.first << ", " << step.second << ")\n";
	//		pathfinding_grid[step.first][step.second] = 2;
	//	}
	// }
	// else
	//{
	//	std::cout << "No path found.\n";
	// }

	//// Display grid for pathfinding algorithm
	// for(size_t y = 0; y < pathfinding_grid.size(); y++)
	//{
	//	for(size_t x = 0; x < pathfinding_grid[y].size() / 2; x++)
	//	{
	//		cout << pathfinding_grid[y][x];
	//	}
	//	cout << endl;
	// }

	vSpawnCumulativeReward.resize(vSpawnPoints.size());
	vSpawnLives.resize(vSpawnPoints.size());
	vSpawnProbabilities.resize(vSpawnPoints.size(), 1);

	spawn_probabilities_distribution = std::discrete_distribution<>(vSpawnProbabilities.begin(), vSpawnProbabilities.end());
	//printf("Adding bots...\n");
	dbg_msg("neuralnetwork", "Adding bots...");

	// auto bot = AddBot("Bot");
	// auto bot_2 = AddBot("Bot2");


	if(count_bots)
	{
		vBotLastPos.resize(count_bots);
		vBotsPath.resize(count_bots);
		vInputInputs.resize(count_bots);
		vInputBlocks.resize(count_bots);
		vOutputs.resize(count_bots);
		vIsPreviouslyHooked.resize(count_bots);
		vPrevHookPos.resize(count_bots);
		vBotsSpawnPos.resize(count_bots);
		//vBotsValidateSpawnPoint.resize(count_bots);
		vBotsLastCheckPoint.resize(count_bots);
		vBotsCumulativeRewards.resize(count_bots);
		vBotBestDistance.resize(count_bots);

		for(size_t i = 0; i < count_bots; i++)
		{
			std::string name = "Bot" + to_string(i);
			auto bot = AddBot(name.c_str());
			int iSpawnPoint = spawn_probabilities_distribution(gen);
			//auto decide_time = time_get_impl();
			auto spawn_point_pos = std::pair<int, int>((int)vSpawnPoints[iSpawnPoint].y / 32, (int)vSpawnPoints[iSpawnPoint].x / 32);
			vBotsPath[i] = astar->findPath(spawn_point_pos, 30);
			//auto now = time_get_impl();
			//cout << "Time to find: " << (float)(now - decide_time) / (float)time_freq() << " Length: " << vBotsPath[i].size() << endl;
			bot->KillCharacter();
			bot->TryRespawn(vSpawnPoints[iSpawnPoint]);
			bot->GetCharacter()->SetSolo(true);
			vBots.push_back(bot);
			vBotsLastCheckPoint[i] = bot->GetCharacter()->m_Pos;
			vBotsSpawnPos[i] = iSpawnPoint;
			//vBotsValidateSpawnPoint[i] = spawn_probabilities_distribution(gen);
			vSpawnLives[iSpawnPoint] += 1;
			vBotBestDistance[i] = {astar->distanceToGoal(spawn_point_pos), 0};
			// auto tr = std::thread(RunNNForward, &model_manager, i, &vEvents, &vFinishEvents, &vInputs, &vOutputs);
			// tr.detach();
			// char aFilename[IO_MAX_PATH_LENGTH];
			// str_format(aFilename, sizeof(aFilename), "%s_%s_%d_%llu.demo", m_aCurrentMap, name.c_str(), m_NetServer.Address().port, time_get());
			// string path_demo = "train/" + dir_name + "/demos/" + aFilename;
			// int ret = m_aDemoRecorder[i].Start(Storage(), m_pConsole, path_demo.c_str(), GameServer()->NetVersion(), m_aCurrentMap, &m_aCurrentMapSha256[MAP_TYPE_SIX], m_aCurrentMapCrc[MAP_TYPE_SIX], "server", m_aCurrentMapSize[MAP_TYPE_SIX], m_apCurrentMapData[MAP_TYPE_SIX]);
		}
	}
	//printf("Bots added\n");
	dbg_msg("neuralnetwork", "Bots added");

	//printf("Initializing neural model...\n");
	dbg_msg("neuralnetwork", "Initializing neural model...");
	model_manager = new ModelManager(count_bots * update_tick / skip_tick, count_bots);
	dbg_msg("neuralnetwork", "Model initialized.");

	dbg_msg("neuralnetwork", "Creating data.csv file for statistics...");
	//printf("Creating data.csv file for statistics...\n");
	{
		char aFilename[IO_MAX_PATH_LENGTH];
		/*std::cout << model_manager.GetLearningRate() << std::endl;
		std::cout << model_manager.GetMiniBatchSize() << std::endl;
		std::cout << model_manager.GetCountPPOEpochs() << std::endl;
		std::cout << count_bots << std::endl;
		std::cout << update_tick << std::endl;*/
		sprintf_s(aFilename, sizeof(aFilename), "lr%.1embs%lldppoe%lldbots%drpb%d.csv", model_manager->GetLearningRate(), model_manager->GetMiniBatchSize(), model_manager->GetCountPPOEpochs(), count_bots, update_tick);
		logger.open("train\\" + dir_name + "\\" + aFilename);
		logger << "Step,Average reward,TPS,Dies,Average distance,Training loss,Actor loss,Critic loss,Learning rate,Time since start,Time to decide,Time to tick,Time rest,Time pre forward,Time forward,Time normal,Time to cpu,Time process last" << endl;
	}
	dbg_msg("neuralnetwork", "data.csv file created and initialized.");

	//printf("data.csv file created and initialized.\n");

	// Recording demo to spectate how model performs
	string path_demo;
	{
		char aFilename[IO_MAX_PATH_LENGTH];
		str_format(aFilename, sizeof(aFilename), "%s_%d_%llu.demo", m_pServer->m_aCurrentMap, m_pServer->m_NetServer.Address().port, time_get());
		path_demo = "train/" + dir_name + "/demos/" + aFilename;
		int ret = m_pServer->m_aDemoRecorder[0].Start(m_pStorage, m_pConsole, path_demo.c_str(), m_pGameContext->NetVersion(), m_pServer->m_aCurrentMap, &m_pServer->m_aCurrentMapSha256[CServer::MAP_TYPE_SIX], m_pServer->m_aCurrentMapCrc[CServer::MAP_TYPE_SIX], "server", m_pServer->m_aCurrentMapSize[CServer::MAP_TYPE_SIX], m_pServer->m_apCurrentMapData[CServer::MAP_TYPE_SIX]);
	}

	ticks_timer = time_get_impl();
	start_ticks = m_pServer->Tick();
	ticks_per_second = 0;

	decide_time = std::chrono::high_resolution_clock::now();
	cumulative_time_to_decide = 0;
	cumulative_time_to_tick = 0;
	cumulative_time_rest = 0;

	// Decide times
	cumulative_time_pre_forward = 0;
	cumulative_time_forward = 0;
	cumulative_time_normal = 0;
	cumulative_time_to_cpu = 0;
	cumulative_time_process_last = 0;

	respawn_all = false;
}

void CNeuralNetwork::PreTick()
{
	if((time_get_impl() - ticks_timer) / time_freq() >= 1.0f)
	{
		ticks_per_second = m_pServer->Tick() - start_ticks;
		// std::cout << "TPS: " << ticks_per_second << std::endl;
		start_ticks = m_pServer->Tick();
		ticks_timer = time_get_impl();
	}
	// Handle bots
	if(m_pServer->Tick() % skip_tick == 0)
	{
		gamelayer = m_pGameContext->Layers()->GameLayer();
		pTiles = static_cast<CTile *>(Kernel()->RequestInterface<IMap>()->GetData(gamelayer->m_Data));
		int map_width = gamelayer->m_Width;
		int map_height = gamelayer->m_Height;
		//printf("6\n");

		//  apply new input
		// decide_time = time_get_impl();
		for(size_t i = 0; i < vBots.size(); i++)
		{
			//printf("HAHHA3.1\n");
			auto bot = vBots[i];
			//printf("HAHHA3.2\n");
			//  Move bot wherever you want
			// vec2 center_coords = vec2(12.f * 32.f, 12.f * 32.f);

			auto bot_character = bot->GetCharacter();
			CCharacterCore *bot_character_core;
			// auto bot_2_character = gamecontext->GetPlayerChar(bot_2->GetCID());
			//printf("HAHHA3.3\n");
			if(respawn_all && model_manager->IsTraining())
			{
				// Add to cumulative spawn distance vector
				// int iOldSpawnPoint = vBotsSpawnPos[i];
				// vSpawnCumulativeDistance[iOldSpawnPoint] += bot_character_core->m_Pos.x - vSpawnPoints[iOldSpawnPoint].x;

				// int iSpawnPoint = (int)round(random_float() * (float)vSpawnPoints.size()) % vSpawnPoints.size();
				int iSpawnPoint;
				/*if(!validating)
				{
					iSpawnPoint = spawn_probabilities_distribution(gen);
				}
				else
				{
					iSpawnPoint = vBotsValidateSpawnPoint[i];
				}*/
				iSpawnPoint = spawn_probabilities_distribution(gen);
				bot->KillCharacter();
				bot->TryRespawn(vSpawnPoints[iSpawnPoint]);

				bot_character = bot->GetCharacter();
				bot_character_core = bot_character->Core();

				bot_character->SetSolo(true);

				vBotsSpawnPos[i] = iSpawnPoint;
				vSpawnLives[iSpawnPoint] += 1;
				vBotsCumulativeRewards[i] = 0;
				bot_character_core->m_Vel = vec2(2.f * random_float() - 1.f, 2.f * random_float() - 1.f);
				vBotsLastCheckPoint[i] = bot_character->m_PrevPos = bot_character->m_Pos = bot_character_core->m_Pos = vSpawnPoints[iSpawnPoint];
				auto spawn_point_pos = std::pair<int, int>((int)vSpawnPoints[iSpawnPoint].y / 32, (int)vSpawnPoints[iSpawnPoint].x / 32);
				vBotBestDistance[i] = {astar->distanceToGoal(spawn_point_pos), m_pServer->Tick()};
				vBotsPath[i] = astar->findPath(spawn_point_pos, 30);
				// bot_2->KillCharacter();
				// bot_2->TryRespawn();
				// bot_character->Core()->m_Pos.x = 3.f * 32.f + random_float() * 4.f * 32.f;
				// bot_2_character->Core()->m_Pos.x = 4.f * 32.f + random_float() * 12.f * 32.f;
				// start_tick = m_CurrentGameTick;
			}

			if(!bot_character)
			{
				continue;
			}

			bot_character_core = bot_character->Core();

			vec2 bot_pos = bot_character->Core()->m_Pos;
			/*std::cout << bot_pos.x
					<< " " << bot_pos.y << std::endl;*/
			// bot_character->Core()->m_HookDir
			// vec2 delta_coords = center_coords - bot_pos;

			/*vec2 rand_pos = {random_float() * 24.f * 32.f, random_float() * 24.f * 32.f};
			delta_coords = center_coords - rand_pos;

			int should_angle = coords_to_angle(delta_coords.x, delta_coords.y);
			int actual_angle = bot_character_core->m_Angle+402;
			prev_angle_dist = calc_angles_distance(actual_angle, should_angle);*/

			ModelInputInputs *input_inputs = &vInputInputs[i];
			ModelInputBlocks *input_blocks = &vInputBlocks[i];

			// auto gamecontext = ((CGameContext *)GameServer());

			// const int Index = (int)(bot_pos.y / 32 + 1) * gamelayer->m_Width + (int)(bot_pos.x / 32);
			// const int GameIndex = pTiles[Index].m_Index;

			input_inputs->pos = {bot_pos.x - (int)bot_pos.x, bot_pos.y - (int)bot_pos.y};
			input_inputs->m_vel = bot_character_core->m_Vel / 20.f;
			input_inputs->is_grounded = bot_character->IsGrounded();

			int UsedJumps = bot_character_core->m_JumpedTotal;
			//cout << "m_Jumps: " << bot_character_core->m_Jumps << endl;
			//cout << "m_JumpedTotal: " << bot_character_core->m_JumpedTotal << endl;
			if(bot_character_core->m_Jumps > 1)
			{
				//cout << "HERE" << endl;
				UsedJumps += !input_inputs->is_grounded;
			}
			input_inputs->can_jump = bot_character->IsGrounded() || (bot_character_core->m_Jumps - UsedJumps);
			input_inputs->is_jumping = vOutputs[i].jump;

			input_inputs->is_hooking = bot_character_core->m_HookState == HOOK_FLYING || bot_character_core->m_HookState == HOOK_GRABBED;
			input_inputs->is_grabbed = bot_character_core->m_HookState == HOOK_GRABBED;
			input_inputs->is_retracted = bot_character_core->m_HookState == HOOK_RETRACTED || (bot_character_core->m_HookState >= HOOK_RETRACT_START && bot_character_core->m_HookState <= HOOK_RETRACT_END);

			if(input_inputs->is_hooking)
			{
				//printf("HJQWHEe1\n");
				auto hook_relative = (bot_character_core->m_HookPos - bot_character_core->m_Pos) / m_pGameContext->Tuning()->m_HookLength;
				//printf("HJQWHEe2\n");
				input_inputs->hook_pos = vec2(clamp(hook_relative.x, -1.f, 1.f), clamp(hook_relative.y, -1.f, 1.f));
				input_inputs->hook_dir = bot_character_core->m_HookDir;
				//printf("HJQWHEe3\n");
				auto ataned = atan2(hook_relative.y, hook_relative.x);
				auto angle_x = cos(ataned);
				auto angle_y = sin(ataned);
				input_inputs->hook_angle = vec2(angle_x, angle_y);
				//printf("HJQWHEe4\n");
				if(vIsPreviouslyHooked[i])
				{
					auto hook_relative_old = vPrevHookPos[i] - bot_character->m_PrevPos;

					ataned = atan2(hook_relative_old.y, hook_relative_old.x);
					angle_x = cos(ataned);
					angle_y = sin(ataned);
					input_inputs->hook_old_angle = vec2(angle_x, angle_y);
				}
				else
				{
					input_inputs->hook_old_angle = vec2(0.f, 0.f);
				}
				//printf("HJQWHEe5\n");

				vIsPreviouslyHooked[i] = true;
				vPrevHookPos[i] = bot_character_core->m_HookPos;
				//printf("HJQWHEe6\n");
			}
			else
			{
				vIsPreviouslyHooked[i] = false;
				input_inputs->hook_pos = input_inputs->hook_dir = input_inputs->hook_angle = input_inputs->hook_old_angle = vec2(0, 0);
			}

			int width = 33;
			int height = 33;
			int block_count = 0;
			//cout << "Starting..." << endl;
			// decide_time = time_get_impl();

			int cur_index_x = std::clamp((int)(bot_pos.x / 32), 0, map_width - 1);
			int cur_index_y = std::clamp((int)(bot_pos.y / 32), 0, map_height - 1);
			int prev_index_x = std::clamp((int)(vBotLastPos[i].x / 32), 0, map_width - 1);
			int prev_index_y = std::clamp((int)(vBotLastPos[i].y / 32), 0, map_height - 1);

			vBotLastPos[i] = bot_pos;

			if(cur_index_x == prev_index_x && cur_index_y == prev_index_y)
			{
				continue;
			}

			//printf("1\n");

			int filled_count = 0;

			for(size_t j = 0; j < vBotsPath[i].size(); j++)
			{
				int x_move = vBotsPath[i][j].second;
				int y_move = vBotsPath[i][j].first;
				// cout << "x: " << x_move << " y: " << y_move << endl;
				input_inputs->path[j] = vec2(x_move, y_move);
				filled_count += 1;
			}
			for(size_t j = filled_count; j < sizeof(ModelInputInputs::path) / (sizeof(float) * 2); j++)
			{
				input_inputs->path[j] = vec2(0.f, 0.f);
			}
			//printf("2\n");

			for(size_t y = 0; y < height; y++)
			{
				for(size_t x = 0; x < width; x++)
				{
					int index_x = std::clamp((int)(bot_pos.x / 32 - width / 2 + x), 0, map_width - 1);
					int index_y = std::clamp((int)(bot_pos.y / 32 - height / 2 + y), 0, map_height - 1);
					int Index = index_y * map_width + index_x;

					/*if(y == height / 2 && width / 2 == x)
					{
						continue;
					}*/
					// input_blocks->blocks[block_count] = pTiles[Index].m_Index;
					//printf("2.1\n");

					switch(pTiles[Index].m_Index)
					{
					case 1:
						input_blocks->blocks[block_count] = pTiles[Index].m_Index;
						break;
					case 9:
						input_blocks->blocks[block_count] = 2;
						break;
					default:
						input_blocks->blocks[block_count] = 0;
						break;
					}
					//printf("2.2\n");

					/*switch(pTiles[Index].m_Index)
					{
					case 1:
						cout << "H";
						break;
					case 2:
						cout << "D";
						break;
					case 3:
						cout << "U";
						break;
					default:
						cout << " ";
						break;
					}*/

					block_count += 1;
				}
				// cout << endl;
			}
			// summerr += time_get_impl() - decide_time;

			//printf("HAHHA3.4\n");
			// vInputs[i] = input;
			//printf("HAHHA3.4.1\n");
			//printf("HAHHA3.5\n");
			//  int64_t decide_time = time_get_impl();
			// ModelOutput returned_model = model_manager.Decide(input); // {0.0f}; //
			//// cout << "Time to decide: " << (float)(time_get_impl() - decide_time) / (float)time_freq() << endl;
			// model_angle = returned_model.angle * 1608;
			// model_direction = returned_model.direction;
			// model_hook = returned_model.hook;
			// started = true;
			//  prev_angle_dist = calc_angles_distance((int)model_angle, should_angle);

			// int should_angle = coords_to_angle(delta_coords.x, delta_coords.y);
			/*actual_angle = bot_character_core->m_Angle + 402;
			int now_dist = calc_angles_distance(model_angle, should_angle);

			float reward = -(((float)now_dist / 402.f) - 1.f);
			model_manager.Reward(reward, (m_CurrentGameTick - start_tick >= 1000) ? 1 : 0);
			rewards.push_back(-calc_angles_distance(model_angle, should_angle));*/

			// model_direction = returned_model.direction;
			// int reward = -calc_angles_distance(model_angle, should_angle);

			/*model_manager.Reward(reward, (m_CurrentGameTick - start_tick >= 1000) ? 1 : 0);
			rewards.push_back(reward);*/

			// std::cout << actual_angle << " " << reward << std::endl;

			/*std::cout << delta_coords.x
					<< " " << delta_coords.y
					<< " Model: " << model_angle
					<< " Should be: " << should_angle
					<< " Reward: " << reward << std::endl;*/
		}
		if(respawn_all)
		{
			respawn_all = false;
		}
		// cout << "Time calcs: " << (float)summerr / (float)time_freq() << endl;

		// auto player_char = gamecontext->GetPlayerChar(c);

		// if(player_char != nullptr)
		//{
		//	auto vel = player_char->Core()->m_Vel;
		//	auto x_pos = player_char->m_Pos.x / 32.0f;
		//	auto y_pos = player_char->m_Pos.y / 32.0f;

		//	for(auto &Input : m_aClients[c].m_aInputs)
		//	{
		//		if(Input.m_GameTick == Tick() + 1)
		//		{
		//			CNetObj_PlayerInput *pApplyInput = (CNetObj_PlayerInput *)Input.m_aData;
		//			auto is_hooking = pApplyInput->m_Hook;
		//			auto direction_moving = pApplyInput->m_Direction;

		//			break;
		//		}
		//	}

		//	const int Index = (int)(y_pos + 1) * gamelayer->m_Width + (int)(x_pos);
		//	const int GameIndex = pTiles[Index].m_Index;
		//	//std::cout << m_aClients[c].m_aName << " " << m_aClients[c].m_Addr.ip[0] << " " << vel.x << std::endl;
		//	//sprintf(buf, "x:%f y:%f %i\n", player_char->m_Pos.x, player_char->m_Pos.y, GameIndex);
		//	//printf(buf);
		//}
		//printf("HAHHA4\n");
		auto now = std::chrono::high_resolution_clock::now();
		cumulative_time_rest += std::chrono::duration_cast<std::chrono::duration<float>>(now - decide_time).count() * 1000.f;
		// cout << "Time rest: " << std::chrono::duration_cast<std::chrono::duration<float>>(now - decide_time).count() << endl;
		double time_pre_forward = 0;
		double time_forward = 0;
		double time_normal = 0;
		double time_to_cpu = 0;
		double time_process_last = 0;
		decide_time = std::chrono::high_resolution_clock::now();
		vOutputs = model_manager->Decide(vInputInputs, vInputBlocks, time_pre_forward, time_forward, time_normal, time_to_cpu, time_process_last);
		now = std::chrono::high_resolution_clock::now();
		cumulative_time_to_decide += std::chrono::duration_cast<std::chrono::duration<float>>(now - decide_time).count() * 1000.f;
		cumulative_time_pre_forward += time_pre_forward;
		cumulative_time_forward += time_forward;
		cumulative_time_normal += time_normal;
		cumulative_time_to_cpu += time_to_cpu;
		cumulative_time_process_last += time_process_last;
		// cout << "Time to decide: " << std::chrono::duration_cast<std::chrono::duration<float>>(now - decide_time).count() << endl;
		decide_time = std::chrono::high_resolution_clock::now();
	}
	decide_time = std::chrono::high_resolution_clock::now();
}

void CNeuralNetwork::PreOnClientPredictedInput()
{
	for(size_t i = 0; i < vBots.size(); i++)
	{
		//printf("HAHHA1111\n");
		auto bot = vBots[i];
		//printf("HAHHA2222\n");
		//  Move bot wherever you want

		// m_Direction:
		// 1 - Right
		// 0 - Stay
		// -1 - Left

		int angle = 0;

		//auto coords = angle_to_coords(angle);

		// angle += 1;
		// angle %= 1608;

		vec2 center_coords = vec2(24.5f * 32.0f, 20.5f * 32.0f);

		auto bot_character = bot->GetCharacter();
		int model_jump = 0;
		//printf("HAHHA2222\n");
		vec2 model_angle;
		int model_direction = 0;
		int model_hook = 0;

		if(bot_character != nullptr)
		{
			// vec2 bot_pos = bot_character->m_Pos;
			/*std::cout << bot_pos.x
				    << " " << bot_pos.y << std::endl;*/

			// vec2 delta_coords = center_coords - bot_pos;

			/*if(bot_character->IsGrounded())
			{
				jump = 1;
			}*/
			ModelOutput returned_model = vOutputs[i]; // {0.0f}; //
			//printf("HAHHA33.2\n");
			//  cout << "Time to decide: " << (float)(time_get_impl() - decide_time) / (float)time_freq() << endl;
			model_angle = returned_model.angle * 299.f;
			model_direction = returned_model.direction;
			model_hook = returned_model.hook;

			if(m_pServer->Tick() % skip_tick != 1)
			{
				model_jump = 0;
			}
			else
			{
				model_jump = returned_model.jump;
			}
			//printf("HAHHA33.3\n");
			//printf("HAHHA34\n");
			//  static ModelManager model_manager;
			//  static std::vector<int> rewards;
			//  static int best_average = -999999;

			// if(m_CurrentGameTick - start_tick >= 1000)
			//{
			//	int average = std::accumulate(rewards.begin(), rewards.end(), 0) / (int)rewards.size();
			//	cout << "Average: " << average << endl;
			//	rewards.clear();

			//	//if(m_aDemoRecorder[c].IsRecording())
			//		//m_aDemoRecorder[c].Stop();

			//	if(average > best_average)
			//	{
			//		best_average = average;
			//		model_manager.Save("best_model_" + to_string(average) + ".pt");

			//		char aNewFilename[IO_MAX_PATH_LENGTH];
			//		str_format(aNewFilename, sizeof(aNewFilename), "demos/average_%d_%s_%s_%llu.demo", average, m_aCurrentMap, m_aClients[c].m_aName, time_get());
			//		//Storage()->RenameFile(m_aDemoRecorder[c].GetCurrentFilename(), aNewFilename, IStorage::TYPE_SAVE);
			//	}
			//	else
			//	{
			//		//char aFilename[IO_MAX_PATH_LENGTH];
			//		//str_format(aFilename, sizeof(aFilename), "demos/%s_%d_%d_tmp.demo", m_aCurrentMap, m_NetServer.Address().port, c);
			//		//Storage()->RemoveFile(m_aDemoRecorder[c].GetCurrentFilename(), IStorage::TYPE_SAVE);
			//	}

			//	//char aFilename[IO_MAX_PATH_LENGTH];
			//	//str_format(aFilename, sizeof(aFilename), "demos/%s_%d_%d_%llu.demo", m_aCurrentMap, m_NetServer.Address().port, c, time_get());
			//	//int ret = m_aDemoRecorder[bot->GetCID()].Start(Storage(), Console(), aFilename, GameServer()->NetVersion(), m_aCurrentMap, &m_aCurrentMapSha256[MAP_TYPE_SIX], m_aCurrentMapCrc[MAP_TYPE_SIX], "server", m_aCurrentMapSize[MAP_TYPE_SIX], m_apCurrentMapData[MAP_TYPE_SIX]);
			//	//printf("ret: %i\n", m_aDemoRecorder[c].IsRecording());

			//	model_manager.Update();
			//}

			// int should_angle = coords_to_angle(delta_coords.x, delta_coords.y);

			// ModelOutput returned_model = model_manager.Decide({bot_pos}); // {0.0f}; //
			// int model_angle = returned_model.angle * 1608;

			// int reward = -calc_angles_distance(model_angle, should_angle);

			// model_manager.Reward(reward, (m_CurrentGameTick - start_tick >= 1000) ? 1 : 0);
			// rewards.push_back(reward);

			// coords = angle_to_coords(model_angle);

			/*std::cout << delta_coords.x
					<< " " << delta_coords.y
				<< " Actual: " << angle
					    << " Should be: " << should_angle
					<< " Reward: " << reward << std::endl;*/
		}

		bot->UpdatePlaytime();

		CNetObj_PlayerInput pApplyInput;
		mem_zero(&pApplyInput, sizeof(pApplyInput));
		pApplyInput.m_TargetX = (int)round(model_angle.x);
		pApplyInput.m_TargetY = (int)round(model_angle.y);
		pApplyInput.m_Jump = model_jump;
		pApplyInput.m_Direction = model_direction;
		pApplyInput.m_Hook = model_hook;

		// auto gamecontext = ((CGameContext *)GameServer());

		// gamecontext->m_apPlayers[0]->m_Score;
		//printf("17\n");
		m_pGameContext->OnClientPredictedInput(bot->GetCID(), &pApplyInput);
		//printf("18\n");
		//  #include <>
		//  if(strcmp(m_aClients[0].m_aName, "heartless tee") == 0)
		//{
		//	for(auto &Input : m_aClients[0].m_aInputs)
		//	{
		//		if(Input.m_GameTick == Tick())
		//		{
		//			CNetObj_PlayerInput pApplyInput;
		//			mem_zero(&pApplyInput, sizeof(pApplyInput));
		//			memcpy(&pApplyInput, Input.m_aData, sizeof(pApplyInput));
		//			GameServer()->OnClientPredictedInput(c, &pApplyInput);
		//			break;
		//		}
		//	}
		//  }
		//  else
		//{
		//	CNetObj_PlayerInput pApplyInput;
		//	mem_zero(&pApplyInput, sizeof(pApplyInput));
		//	pApplyInput.m_TargetX = coords.x;
		//	pApplyInput.m_TargetY = coords.y;

		//	//auto gamecontext = ((CGameContext *)GameServer());

		//	// gamecontext->m_apPlayers[0]->m_Score;

		//	GameServer()->OnClientPredictedInput(c, &pApplyInput);
		//}
	}
}

void CNeuralNetwork::PreOnClientPredictedEarlyInput()
{}

void CNeuralNetwork::PostTick(float time_to_tick)
{
	static int dies = 0;
	static int moved_distance = 0;
	//static int validating_moved_distance = 0;
	static float cumulative_reward = 0;
	static int count_updated = 0;

	// Rewards
	static float checkpoint_reward = 100.f / 32.f;
	static float die_reward = -250.f / 32.f; // -500.f / 32.f
	static float finish_reward = 1000.f / 32.f;
	static float step_reward = -0.01f;
	static int long_no_improvements = 200;

	static std::vector<float> rewards;
	static float best_average = -999999.f;
	static float last_saved = -999999.f;

	auto now = std::chrono::high_resolution_clock::now();
	cumulative_time_to_tick += time_to_tick * 1000.f;

	//printf("HAHHA1\n");
	// size_t summerr = 0;
	// decide_time = time_get_impl();
	if(m_pServer->Tick() % skip_tick == 0)
	{
		for(size_t i = 0; i < vBots.size(); i++)
		{
			auto bot = vBots[i];
			// Move bot wherever you want
			// vec2 center_coords = vec2(12.f * 32.f, 12.f * 32.f);
			//printf("8\n");
			auto bot_character = bot->GetCharacter();
			//printf("9\n");
			//  auto bot_2_character = gamecontext->GetPlayerChar(bot_2->GetCID());
			bool died = false;
			bool finished = false;
			bool freezed = false;

			if(bot_character != nullptr)
			{
				vec2 bot_pos = bot_character->Core()->m_Pos;

				int bot_block_pos_x = (int)(bot_pos.x / 32);
				int bot_block_pos_y = (int)(bot_pos.y / 32);
				int bot_block_index = bot_block_pos_y * gamelayer->m_Width + bot_block_pos_x;

				if(pTiles[bot_block_index].m_Index == 34)
				{
					finished = true;
				}

				if(bot_character->m_FreezeTime || bot_character->Core()->m_IsInFreeze || pTiles[bot_block_index].m_Index == 9 || m_pServer->Tick() - vBotBestDistance[i].second >= long_no_improvements)
				{
					freezed = true;
				}
			}

			if((bot_character == nullptr) || freezed || finished)
			{
				if(bot_character == nullptr || freezed)
				{
					died = true;
					dies += 1;
					if(bot_character)
					{
						bot->KillCharacter();
					}

					/*m_aDemoRecorder[bot->GetCID()].Stop();

					Storage()->RemoveFile(m_aDemoRecorder[bot->GetCID()].GetCurrentFilename(), IStorage::TYPE_ABSOLUTE);

					char aweweFilename[IO_MAX_PATH_LENGTH];
					str_format(aweweFilename, sizeof(aweweFilename), "%llu_%s_%d_%llu.demo", i, m_aCurrentMap, m_NetServer.Address().port, time_get_impl());
					path_demo = "train/" + dir_name + "/demos/" + aweweFilename;
					int ret = m_aDemoRecorder[bot->GetCID()].Start(Storage(), m_pConsole, path_demo.c_str(), GameServer()->NetVersion(), m_aCurrentMap, &m_aCurrentMapSha256[MAP_TYPE_SIX], m_aCurrentMapCrc[MAP_TYPE_SIX], "server", m_aCurrentMapSize[MAP_TYPE_SIX], m_apCurrentMapData[MAP_TYPE_SIX]);*/
				}
				else if(finished)
				{
					/*m_aDemoRecorder[bot->GetCID()].Stop();

					char aNewFilename[IO_MAX_PATH_LENGTH];
					str_format(aNewFilename, sizeof(aNewFilename), "time_%.2f_%llu.demo", (float)(m_CurrentGameTick - bot->GetCharacter()->m_StartTime) / 50.f, time_get_impl());
					path_demo = "train/" + dir_name + "/demos/" + aNewFilename;
					Storage()->RenameFile(m_aDemoRecorder[bot->GetCID()].GetCurrentFilename(), path_demo.c_str(), IStorage::TYPE_ABSOLUTE);*/

					bot->KillCharacter();

					/*char aFilename[IO_MAX_PATH_LENGTH];
					str_format(aFilename, sizeof(aFilename), "%llu_%s_%d_%llu.demo", i, m_aCurrentMap, m_NetServer.Address().port, time_get_impl());
					path_demo = "train/" + dir_name + "/demos/" + aFilename;
					int ret = m_aDemoRecorder[bot->GetCID()].Start(m_pStorage, m_pConsole, path_demo.c_str(), m_pGameContext->NetVersion(), m_pServer->m_aCurrentMap, &m_pServer->m_aCurrentMapSha256[CServer::MAP_TYPE_SIX], m_pServer->m_aCurrentMapCrc[CServer::MAP_TYPE_SIX], "server", m_pServer->m_aCurrentMapSize[CServer::MAP_TYPE_SIX], m_pServer->m_apCurrentMapData[CServer::MAP_TYPE_SIX]);*/

					/*char aFilename[IO_MAX_PATH_LENGTH];
					str_format(aFilename, sizeof(aFilename), "%s_%s_%d_%llu.demo", m_aCurrentMap, name.c_str(), m_NetServer.Address().port, time_get());
					string path_demo = "train/" + dir_name + "/demos/" + aFilename;
					int ret = m_aDemoRecorder[i].Start(Storage(), m_pConsole, path_demo.c_str(), GameServer()->NetVersion(), m_aCurrentMap, &m_aCurrentMapSha256[MAP_TYPE_SIX], m_aCurrentMapCrc[MAP_TYPE_SIX], "server", m_aCurrentMapSize[MAP_TYPE_SIX], m_apCurrentMapData[MAP_TYPE_SIX]);*/
				}
				//if(validating)
				//{
				//	validating_dones += 1;
				//	if(validating_dones == count_bots)
				//	{
				//		validated = true;
				//		validating = false;
				//		//printf("Validated\n");
				//		break;
				//	}
				//}

				// decide_time = time_get_impl();
				// bot_character->Core()->m_IsInFreeze
				// int iSpawnPoint = (int)round(random_float() * (float)vSpawnPoints.size()) % vSpawnPoints.size();
				int iSpawnPoint = spawn_probabilities_distribution(gen);
				bot->TryRespawn(vSpawnPoints[iSpawnPoint]);
				// summerr += time_get_impl() - decide_time;
				bot_character = bot->GetCharacter();
				bot_character->SetSolo(true);
				if(bot_character == nullptr)
				{
					dbg_msg("neuralnetwork", "FFFFUUUUUCCCCKKK");
					//cout << "FFFFUUUUUCCCCKKK" << endl;
					exit(1);
				}

				vBotsSpawnPos[i] = iSpawnPoint;
				vSpawnLives[iSpawnPoint] += 1;
				vBotsLastCheckPoint[i] = bot_character->m_PrevPos = bot_character->m_Pos = bot_character->Core()->m_Pos = vSpawnPoints[iSpawnPoint];
				bot_character->Core()->m_Vel = vec2(2.f * random_float() - 1.f, 2.f * random_float() - 1.f);
				vBotsCumulativeRewards[i] = 0;
				auto spawn_point_pos = std::pair<int, int>((int)vSpawnPoints[iSpawnPoint].y / 32, (int)vSpawnPoints[iSpawnPoint].x / 32);
				vBotsPath[i] = astar->findPath(spawn_point_pos, 30);
				vBotBestDistance[i] = {astar->distanceToGoal(spawn_point_pos), m_pServer->Tick()};
				//printf("respawn: %d %d %d\n", spawn_point_pos.first, spawn_point_pos.second, vBotBestDistance[i].first);

				// bot_character->Core()->m_Pos.x = 3.f * 32.f + random_float() * 4.f * 32.f;
			}

			if(bot_character)
			{
				float reward = 0.f;

				vec2 bot_pos = bot_character->Core()->m_Pos;

				int bot_block_pos_x = (int)(bot_pos.x / 32);
				int bot_block_pos_y = (int)(bot_pos.y / 32);
				int bot_block_index = bot_block_pos_y * gamelayer->m_Width + bot_block_pos_x;

				if(died)
				{
					reward += die_reward; // 1000 500 348 * 32;
				}
				else if(finished)
				{
					reward += finish_reward; // 348 * 32;
				}
				else
				{
					// If touched the checkpoint reward
					if(pTiles[bot_block_index].m_Index == 35 && vBotsLastCheckPoint[i].x < bot_block_pos_x * 32.f)
					{
						reward += checkpoint_reward; // 20 50 96 * 32;
						vBotsLastCheckPoint[i] = vec2(bot_block_pos_x * 32.f, bot_block_pos_y * 32.f);
					}
					// reward += bot_character->m_Pos.x - bot_character->m_PrevPos.x;
					int bot_last_block_pos_x = vBotLastPos[i].x / 32;
					int bot_last_block_pos_y = vBotLastPos[i].y / 32;
					if(bot_last_block_pos_x != bot_block_pos_x || bot_last_block_pos_y != bot_block_pos_y)
					{
						int prev_dist = astar->distanceToGoal(bot_last_block_pos_y, bot_last_block_pos_x);
						vBotsPath[i] = astar->findPath(std::pair<int, int>(bot_block_pos_y, bot_block_pos_x), 30);
						int current_dist = astar->distanceToGoal(bot_block_pos_y, bot_block_pos_x);
						int path_dist_diff = prev_dist - current_dist;
						reward += path_dist_diff;
						moved_distance += path_dist_diff;

						//printf("%d  %d\n", current_dist, vBotBestDistance[i].first);
						if(current_dist < vBotBestDistance[i].first)
						{
							vBotBestDistance[i] = {current_dist, m_pServer->Tick()};
						}
					}

					auto tick_diff = m_pServer->Tick() - vBotBestDistance[i].second;

					auto long_stay_penalty = -0.0003f * tick_diff;

					// Encourage to run faster
					reward += step_reward; // step_reward

					/*if(bot_character->m_Pos.x < 10.f * 32.f && reward <= 0)
					{
						reward = -32 * 32;
					}*/
				}

				// Add to cumulative spawn distance vector
				int iOldSpawnPoint = vBotsSpawnPos[i];
				vSpawnCumulativeReward[iOldSpawnPoint] += reward;
				cumulative_reward += reward;

				// int prev_dist = (int)abs(bot_character->m_PrevPos.x - bot_2_character->m_PrevPos.x);
				// int now_dist = (int)abs(bot_character->m_Pos.x - bot_2_character->m_Pos.x);
				// reward += prev_dist - now_dist;
				// model_manager.Reward(reward, (m_CurrentGameTick - start_tick >= 1000) ? 1 : 0);
				// rewards.push_back(-(int)abs(bot_character->m_Pos.x - bot_2_character->m_Pos.x));

				/*vec2 delta_coords = center_coords - bot_character->m_Pos;
				int should_angle = coords_to_angle(delta_coords.x, delta_coords.y);
				int actual_angle = bot_character->Core()->m_Angle + 402;
				int now_dist = calc_angles_distance(actual_angle, should_angle);

				reward += prev_angle_dist - now_dist;
				model_manager.Reward(reward, (m_CurrentGameTick - start_tick >= 1000) ? 1 : 0);
				rewards.push_back(-calc_angles_distance(actual_angle, should_angle));*/

				bool is_done = died || finished /*|| (m_CurrentGameTick % update_tick == 0) ? 1 : 0*/;
				model_manager->Reward(reward, is_done);
				if(!died && !finished)
				{
					vBotsCumulativeRewards[i] += reward;
				}
				char reward_name[16];
				sprintf_s(reward_name, "%.1f %d", vBotsCumulativeRewards[i], vBotBestDistance[i].first);
				memcpy(m_pServer->m_aClients[bot->GetCID()].m_aName, reward_name, strlen(reward_name) + 1);
				// std::cout << reward << std::endl;
				rewards.push_back(reward);
			}
		}
		bool is_full = false;
		// cout << "Time rewards: " << (float)(time_get_impl() - decide_time) / (float)time_freq() << endl;
		model_manager->SaveReplays(is_full);
		// cout << m_CurrentGameTick << endl;
		ticks_collected += 1;

		if(is_full)
		{
			// decide_time = time_get_impl();
			//printf("UPDATING\n");
			//printf("Updating.\n");

			float avg_reward = cumulative_reward / (float)(dies + count_bots);
			float avg_dist = ((float)moved_distance / (float)(dies));
			//float avg_valid_dist = (float)validating_moved_distance / (float)count_bots;
			//rewards.clear();

			auto demo_recorder = &m_pServer->m_aDemoRecorder[0];

			if(demo_recorder->IsRecording() && model_manager->IsTraining())
			{
				demo_recorder->Stop();
				char aNewFilename[IO_MAX_PATH_LENGTH];
				str_format(aNewFilename, sizeof(aNewFilename), "average_dist_%.2f_rew_%.2f_%s_%llu.demo", avg_dist, avg_reward, m_pServer->m_aCurrentMap, time_get_impl());
				string path_demo = "train/" + dir_name + "/demos/" + aNewFilename;
				m_pStorage->RenameFile(demo_recorder->GetCurrentFilename(), path_demo.c_str(), IStorage::TYPE_ABSOLUTE);
			}
			//printf("111\n");
			/*if(ticks_collected % (count_ticks * 20) == 0 && model_manager->IsTraining())
			{
				model_manager->Save("train\\" + dir_name + "\\models\\last");
			}*/
			//printf("222\n");
			//if(avg_dist > best_average && model_manager->IsTraining())
			//{
			//	best_average = avg_dist;
			//	model_manager->Save("train\\" + dir_name + "\\models\\best"); // best" + to_string(average)

			//	/*if(was_recording)
			//	{
			//		char aNewFilename[IO_MAX_PATH_LENGTH];
			//		str_format(aNewFilename, sizeof(aNewFilename), "average_%f_%s_%llu.demo", average, m_aCurrentMap, time_get());
			//		path_demo = "train/" + dir_name + "/demos/" + aNewFilename;
			//		Storage()->RenameFile(demo_recorder->GetCurrentFilename(), path_demo.c_str(), IStorage::TYPE_ABSOLUTE);
			//	}*/
			//}
			//else
			//{
			//	/*if(was_recording)
			//	{
			//		Storage()->RemoveFile(demo_recorder->GetCurrentFilename(), IStorage::TYPE_ABSOLUTE);
			//	}*/
			//}
			//printf("ret: %i\n", ret);
			//printf("start_u\n");
			// int64_t update_time = time_get_impl();
			double avg_training_loss = 0;
			double avg_actor_loss = 0;
			double avg_critic_loss = 0;
			bool updated = false;
			size_t count_episodes = model_manager->GetCountEpisodes();
			static size_t count_episodes_processed = 0;
			static size_t count_every_update = 0;
			model_manager->Update(avg_dist, dies, updated, avg_training_loss, avg_actor_loss, avg_critic_loss);
			count_episodes_processed += count_episodes;
			count_every_update += 1;
			// cout << "Time update: " << (float)(time_get_impl() - decide_time) / (float)time_freq() << endl;
			if(updated)
			{
				count_updated += 1;
				rewards.clear();
				if(avg_dist - last_saved > 25)
				{
					if(!m_pStorage->CpyFile(("train\\" + dir_name + "\\models\\last_model.pt").c_str(), ("train\\" + dir_name + "\\models\\early_stopping_" + to_string(avg_dist) + "_model.pt").c_str(), false))
					{
						dbg_msg("neuralnetwork", "Failed to copy early stopped model");
					}
					if(!m_pStorage->CpyFile(("train\\" + dir_name + "\\models\\last_optimizer.pt").c_str(), ("train\\" + dir_name + "\\models\\early_stopping_" + to_string(avg_dist) + "_optimizer.pt").c_str(), false))
					{
						dbg_msg("neuralnetwork", "Failed to copy early stopped optimizer");
					}
					//model_manager->Save("train\\" + dir_name + "\\models\\early_stopping_" + to_string(avg_dist));
					last_saved = avg_dist;
				}

				if(avg_dist > best_average && model_manager->IsTraining())
				{
					best_average = avg_dist;
					if(!m_pStorage->CpyFile(("train\\" + dir_name + "\\models\\last_model.pt").c_str(), ("train\\" + dir_name + "\\models\\best_model.pt").c_str(), false))
					{
						dbg_msg("neuralnetwork", "Failed to copy best model");
					}
					if(!m_pStorage->CpyFile(("train\\" + dir_name + "\\models\\last_optimizer.pt").c_str(), ("train\\" + dir_name + "\\models\\best_optimizer.pt").c_str(), false))
					{
						dbg_msg("neuralnetwork", "Failed to copy best optimizer");
					}

					//model_manager->Save("train\\" + dir_name + "\\models\\best"); // best" + to_string(average)

					/*if(was_recording)
					{
						char aNewFilename[IO_MAX_PATH_LENGTH];
						str_format(aNewFilename, sizeof(aNewFilename), "average_%f_%s_%llu.demo", average, m_aCurrentMap, time_get());
						path_demo = "train/" + dir_name + "/demos/" + aNewFilename;
						Storage()->RenameFile(demo_recorder->GetCurrentFilename(), path_demo.c_str(), IStorage::TYPE_ABSOLUTE);
					}*/
				}
				else
				{
					/*if(was_recording)
					{
						Storage()->RemoveFile(demo_recorder->GetCurrentFilename(), IStorage::TYPE_ABSOLUTE);
					}*/
				}

				model_manager->Save("train\\" + dir_name + "\\models\\last");

				if(count_updated % 20 == 0 && model_manager->IsTraining())
				{
					dbg_msg("neuralnetwork", "UPDATING");
					std::vector<float> vAverageDistancePerSpawn(vSpawnCumulativeReward.size());
					int count_counted = 0;
					float cumulative_reward = 0;
					//printf("1\n");
					for(size_t i = 0; i < vSpawnCumulativeReward.size(); i++)
					{
						if(vSpawnLives[i] && vSpawnCumulativeReward[i] != 0.f)
						{
							float dist = vSpawnCumulativeReward[i] / (float)vSpawnLives[i];
							cumulative_reward += dist;
							vAverageDistancePerSpawn[i] = dist;
							count_counted += 1;
						}
					}
					//printf("1.9\n");
					float average_reward = 0;
					if(cumulative_reward != 0.f && count_counted)
					{
						average_reward = cumulative_reward / (float)count_counted;
					}
					dbg_msg("neuralnetwork", "Average reward: %f", average_reward);
					//cout << "Average reward: " << average_reward << endl;
					//printf("1.99\n");
					float max_reward = *max_element(vAverageDistancePerSpawn.begin(), vAverageDistancePerSpawn.end());
					dbg_msg("neuralnetwork", "Max reward: %f", max_reward);
					//cout << "Max reward: " << max_reward << endl;
					//printf("2\n");
					for(size_t i = 0; i < vSpawnCumulativeReward.size(); i++)
					{
						//printf("2.1\n");
						if(!vSpawnLives[i] || !vSpawnCumulativeReward[i])
						{
							//printf("2.1.1\n");
							vAverageDistancePerSpawn[i] = average_reward;
						}
						//printf("2.2\n");
						vSpawnProbabilities[i] = max_reward - vAverageDistancePerSpawn[i] + 1;
						//printf("2.3\n");
						vSpawnLives[i] = 0;
						//printf("2.4\n");
						vSpawnCumulativeReward[i] = 0;
					}
					//printf("2.9\n");
					spawn_probabilities_distribution = std::discrete_distribution<>(vSpawnProbabilities.begin(), vSpawnProbabilities.end());
					//printf("2.10\n");
					/*for(size_t i = 0; i < vSpawnProbabilities.size(); i++)
					{
						cout << vSpawnProbabilities[i] << endl;
					}*/
				}
				logger << ticks_collected / count_ticks
					    << "," << avg_reward
					    << "," << ticks_per_second
					    << "," << dies
					    << "," << avg_dist
					    //<< "," << avg_valid_dist
					    << "," << avg_training_loss
					    << "," << avg_actor_loss
					    << "," << avg_critic_loss
					    << "," << model_manager->GetCurrentLearningRate()
					    << "," << (float)time_get_impl() / (float)time_freq()
					    << "," << (cumulative_time_to_decide / (float)update_tick)
					    << "," << (cumulative_time_to_tick / (float)update_tick)
					    << "," << (cumulative_time_rest / (float)update_tick)
					    << "," << (cumulative_time_pre_forward / (float)update_tick)
					    << "," << (cumulative_time_forward / (float)update_tick)
					    << "," << (cumulative_time_normal / (float)update_tick)
					    << "," << (cumulative_time_to_cpu / (float)update_tick)
					    << "," << (cumulative_time_process_last / (float)update_tick)
					    << endl;
				dbg_msg("neuralnetwork", "Avg. reward: %f TPS: %d Avg. Training Loss: %f Dies: %d Episodes: %d/%d Updates: %d Avg. distance: %f", avg_reward, ticks_per_second, avg_training_loss, dies, count_episodes, count_episodes_processed, count_every_update, avg_dist);
					
				/*cout << "Avg. reward: " << avg_reward << " TPS: " << ticks_per_second << " Avg. Training Loss: " << avg_training_loss
					    << " Dies: " << dies << " Episodes: " << count_episodes << "/" << count_episodes_processed << " Updates: " << count_every_update
					    << " Avg. distance: " << avg_dist << " Avg. valid distance: " << avg_valid_dist << endl;*/
				dies = moved_distance = cumulative_reward = cumulative_time_to_decide = cumulative_time_to_tick = cumulative_time_rest = cumulative_time_pre_forward = cumulative_time_forward = cumulative_time_normal = cumulative_time_normal = cumulative_time_to_cpu = cumulative_time_process_last = count_episodes_processed = count_every_update = 0;
			}
			respawn_all = true;
			// cout << "Time to update: " << (float)(time_get_impl() - update_time) / (float)time_freq() << endl;
			//printf("end\n");
			if(updated && count_updated % 20 == 0 && model_manager->IsTraining())
			{
				char aFilename[IO_MAX_PATH_LENGTH];
				str_format(aFilename, sizeof(aFilename), "%s_%d_%llu.demo", m_pServer->m_aCurrentMap, m_pServer->m_NetServer.Address().port, time_get());
				string path_demo = "train\\" + dir_name + "\\demos\\" + aFilename;
				int ret = demo_recorder->Start(m_pStorage, m_pConsole, path_demo.c_str(), m_pGameContext->NetVersion(), m_pServer->m_aCurrentMap, &m_pServer->m_aCurrentMapSha256[CServer::MAP_TYPE_SIX], m_pServer->m_aCurrentMapCrc[CServer::MAP_TYPE_SIX], "server", m_pServer->m_aCurrentMapSize[CServer::MAP_TYPE_SIX], m_pServer->m_apCurrentMapData[CServer::MAP_TYPE_SIX]);
			}
			decide_time = std::chrono::high_resolution_clock::now();
		}
	}

	//printf("21\n");
	/*auto now = std::chrono::high_resolution_clock::now();
	cumulative_time_to_tick += std::chrono::duration_cast<std::chrono::duration<float>>(now - decide_time).count() * 1000.f;*/
	//cout << "Time to tick: " << (float)(now - decide_time) / (float)time_freq() << endl;
	decide_time = std::chrono::high_resolution_clock::now();

	/*if(m_CurrentGameTick == 1000)
	{
		m_aDemoRecorder[MAX_CLIENTS-1].Stop();
		exit(0);
	}*/
}