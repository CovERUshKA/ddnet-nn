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
#include <game/server/gamemodes/DDRace.h>

#include <numeric>
//#include <iostream>

using namespace std;

CServer *m_pServer;
IStorage *m_pStorage;
IConsole *m_pConsole;
CGameContext *m_pGameContext;
CCollision *m_pCollision;
CGameControllerDDRace *m_pController;

float volleyball_net_height = 9 * 32.f;

vec2 left_spawn_pos = {162.5 * 32.f, 100.5 * 32.f};
vec2 right_spawn_pos = {184.5 * 32.f, 100.5 * 32.f};
vec2 ball_spawn_pos = {173.5 * 32.f, 110.5 * 32.f};

vec2 volleyball_area_start = {160 * 32.f, 98 * 32.f};
vec2 volleyball_area_end = {187 * 32.f, 119 * 32.f};
vec2 volleyball_area_sizes = volleyball_area_end - volleyball_area_start;
vec2 volleyball_area_center = (volleyball_area_start + volleyball_area_end) / 2.f;
//vec2 volleyball_area_left_side = (volleyball_area_start + volleyball_area_end) / 2.f;
//vec2 volleyball_area_right_side = (volleyball_area_start + volleyball_area_end) / 2.f;

vec2 volleyball_dividing_line_start = {volleyball_area_center.x, volleyball_area_start.y};
vec2 volleyball_dividing_line_end = {volleyball_area_center.x, volleyball_area_start.y + volleyball_area_sizes.y - volleyball_net_height};

vec2 volleyball_ball_left_side_spawn = {162, 116};
vec2 volleyball_ball_right_side_spawn = {184, 116};
vec2 volleyball_ball_center_spawn = {173, 110};

vec2 volleyball_final_area_right_top_corner = {67 * 32.f, 60 * 32.f};

int GetFirstEmptySlotId()
{
	for(int ClientID = 0; ClientID < MAX_CLIENTS; ClientID++)
	{
		if(m_pServer->m_aClients[ClientID].m_State == CServer::CClient::STATE_EMPTY)
			return ClientID;
	}

	return -1;
}

bool GetEnemyAndBallId(int ClientID, int Team, int& enemy_id, int& ball_id) {
	int count_found = 0;
	for (int i = 0; i < MAX_CLIENTS && count_found < 2; ++i)
	{
		if(i != ClientID && m_pController->m_Teams.m_Core.Team(i) == Team)
		{
			CCharacter *character = m_pGameContext->m_apPlayers[i]->GetCharacter();
			if(character->Core()->m_DeepFrozen)
				ball_id = i;
			else
				enemy_id = i;
			count_found += 1;
		}
	}
			
	return count_found == 2;
}

//CPlayer *CNeuralNetwork::GetEmptyTeam(int ClientID)
//{
//	return m_pGameContext->m_apPlayers[ClientID];
//}

CPlayer *CNeuralNetwork::AddBot(std::string name, vec2 spawn_pos = {0,0})
{
	for(int ClientID = 0; ClientID < MAX_CLIENTS; ClientID++)
	{
		if(m_pServer->m_aClients[ClientID].m_State == CServer::CClient::STATE_EMPTY)
		{
			// m_aClients[ClientID].m_aName = "Bot";
			memcpy(m_pServer->m_aClients[ClientID].m_aName, name.c_str(), name.length() + 1);

			m_pServer->m_aClients[ClientID].m_State = CServer::CClient::STATE_INGAME;
			m_pServer->m_aClients[ClientID].m_SnapRate = CServer::CClient::SNAPRATE_FULL;
			m_pServer->m_aClients[ClientID].m_DDNetVersion = VERSION_DDRACE;
			m_pServer->m_aClients[ClientID].m_pRconCmdToSend = 0;

			m_pGameContext->OnClientConnected(ClientID, nullptr);
			m_pGameContext->OnClientEnter(ClientID);

			CPlayer *pPlayer = m_pGameContext->m_apPlayers[ClientID];
			pPlayer->SetAfk(false);
			pPlayer->TryRespawn(spawn_pos);

			return pPlayer;
		}
	}

	return nullptr;
}

//CPlayer *CNeuralNetwork::DisconnectBot(int ClientID, vec2 spawn_pos = {0, 0})
//{
//	for(int ClientID = 0; ClientID < MAX_CLIENTS; ClientID++)
//	{
//		if(m_pServer->m_aClients[ClientID].m_State == CServer::CClient::STATE_EMPTY)
//		{
//			// m_aClients[ClientID].m_aName = "Bot";
//			memcpy(m_pServer->m_aClients[ClientID].m_aName, name, strlen(name) + 1);
//
//			m_pServer->m_aClients[ClientID].m_State = CServer::CClient::STATE_INGAME;
//			m_pServer->m_aClients[ClientID].m_SnapRate = CServer::CClient::SNAPRATE_FULL;
//			m_pServer->m_aClients[ClientID].m_DDNetVersion = VERSION_DDRACE;
//			m_pServer->m_aClients[ClientID].m_pRconCmdToSend = 0;
//
//			m_pGameContext->OnClientConnected(ClientID, nullptr);
//			m_pGameContext->OnClientEnter(ClientID);
//
//			CPlayer *pPlayer = m_pGameContext->m_apPlayers[ClientID];
//			pPlayer->SetAfk(false);
//			pPlayer->TryRespawn(spawn_pos);
//
//			return pPlayer;
//		}
//	}
//
//	return nullptr;
//}

// Function to calculate distance from point to segment using float
float ClosestDistanceToDividingLine(vec2 ball_pos)
{
	// Length of a segment squared
	float segment_length_squared = powf(volleyball_dividing_line_end.x - volleyball_dividing_line_start.x, 2) + powf(volleyball_dividing_line_end.y - volleyball_dividing_line_start.y, 2);

	// If a segment degenerates into a point
	if(segment_length_squared == 0)
	{
		return sqrtf(powf(ball_pos.x - volleyball_dividing_line_start.x, 2) + powf(ball_pos.y - volleyball_dividing_line_start.y, 2));
	}

	// Projection of a point onto a segment with normalization from 0 to 1
	float t = max(0.0f, min(1.0f, ((ball_pos.x - volleyball_dividing_line_start.x) * (volleyball_dividing_line_end.x - volleyball_dividing_line_start.x) + (ball_pos.y - volleyball_dividing_line_start.y) * (volleyball_dividing_line_end.y - volleyball_dividing_line_start.y)) / segment_length_squared));

	// Coordinates of the projection of a point onto a segment
	float projection_x = volleyball_dividing_line_start.x + t * (volleyball_dividing_line_end.x - volleyball_dividing_line_start.x);
	float projection_y = volleyball_dividing_line_start.y + t * (volleyball_dividing_line_end.y - volleyball_dividing_line_start.y);

	// Distance from point to projection
	return sqrtf(powf(ball_pos.x - projection_x, 2) + powf(ball_pos.y - projection_y, 2));
}

// team starts from 1
// Number should be exact number like in game
bool CNeuralNetwork::IsSwitchEnabled(int Number, int Team)
{
	auto *switcher = &m_pGameContext->Switchers()[Number];
	return switcher->m_aStatus[Team];
}

// team starts from 1
// Number should be exact number like in game
void CNeuralNetwork::ChangeSwitchState(int Number, int Team, bool state)
{
	auto *switcher = &m_pGameContext->Switchers()[Number];

	switcher->m_aStatus[Team] = state;
	switcher->m_aEndTick[Team] = 0;
	switcher->m_aType[Team] = state ? TILE_SWITCHOPEN : TILE_SWITCHCLOSE;
	switcher->m_aLastUpdateTick[Team] = m_pServer->Tick();
}

/// team starts from 1
// Number should be exact number like in game
void CNeuralNetwork::RespawnTeam(int Team)
{
	int ball_id = (Team - 1) * 3 + 2;
	for(size_t i = 0; i < 3; i++)
	{
		auto bot = vBots[(Team - 1) * 3 + i];

		bot->KillCharacter();

		vec2 spawn_pos;

		if(i % 3 == 2)
		{
			spawn_pos = ball_spawn_pos;
		}
		else
		{
			spawn_pos = i % 3 == 0 ? left_spawn_pos : right_spawn_pos;

			float rand_are_spawn = random_float();

			if(rand_are_spawn > 0.6f)
			{
				spawn_pos = volleyball_area_start;
				spawn_pos.y += 1.5f * 32.f;
				if(rand_are_spawn > 0.9)
					spawn_pos.x += i % 3 == 0 ? 15.5f * 32.f : 1.5f * 32.f;
				else
					spawn_pos.x += i % 3 == 0 ? 1.5f * 32.f : 15.5f * 32.f;

				spawn_pos.x += 10.f * 32.f * random_float();
				spawn_pos.y += 15.f * 32.f * random_float();
			}
		}
		bot->TryRespawn(spawn_pos);
		if(m_pController->m_Teams.SetCharacterTeam(bot->GetCID(), Team) != nullptr)
		{
			std::cout << "Fuck setting team" << std::endl;
		}
		bot->GetCharacter()->SetActiveWeapon(WEAPON_HAMMER);
		bot->GetCharacter()->m_TeleCheckpoint = i % 3 + 1;

		/*if (!model_manager->IsTraining())
		{
			char aFilename[IO_MAX_PATH_LENGTH];
			str_format(aFilename, sizeof(aFilename), "%s_%s_%d_%llu.demo", m_pServer->m_aCurrentMap, name.c_str(), m_pServer->m_NetServer.Address().port, time_get());
			string path_demo = "train/" + dir_name + "/demos/" + aFilename;
			int ret = m_pServer->m_aDemoRecorder[i].Start(m_pStorage, m_pConsole, path_demo.c_str(), m_pGameContext->NetVersion(), m_pServer->m_aCurrentMap, &m_pServer->m_aCurrentMapSha256[CServer::MAP_TYPE_SIX], m_pServer->m_aCurrentMapCrc[CServer::MAP_TYPE_SIX], "server", m_pServer->m_aCurrentMapSize[CServer::MAP_TYPE_SIX], m_pServer->m_apCurrentMapData[CServer::MAP_TYPE_SIX]);
		}*/
	}
	m_pController->m_Teams.ResetRoundState(Team);
	// Start
	ChangeSwitchState(1, Team, true);
	ChangeSwitchState(2, Team, true);
	ChangeSwitchState(3, Team, true);

	// Ball falling for score mode
	ChangeSwitchState(13, Team, false);
	ChangeSwitchState(10, Team, true);
	ChangeSwitchState(11, Team, true);
	ChangeSwitchState(12, Team, true);
	ChangeSwitchState(14, Team, true);
	ChangeSwitchState(15, Team, true);
	ChangeSwitchState(16, Team, true);
	ChangeSwitchState(20, Team, false);
	ChangeSwitchState(17, Team, true);
	ChangeSwitchState(18, Team, true);
	ChangeSwitchState(19, Team, true);
	ChangeSwitchState(21, Team, true);
	ChangeSwitchState(22, Team, true);
	ChangeSwitchState(23, Team, true);
	ChangeSwitchState(24, Team, true);
	ChangeSwitchState(32, Team, true);

	vBots[ball_id]->GetCharacter()->SetDeepFrozen(true);
}

void CNeuralNetwork::StartFight(CPlayer* player, bool right_side)
{
	int Team = m_pController->m_Teams.GetFirstEmptyTeam();

	// Handle player
	player->KillCharacter();
	player->TryRespawn(right_side ? right_spawn_pos : left_spawn_pos);
	m_pController->m_Teams.SetCharacterTeam(player->GetCID(), Team);
	player->GetCharacter()->m_TeleCheckpoint = right_side ? 2 : 1;
	player->GetCharacter()->Core()->m_Jumps = 0;
	//printf("Spawning bot\n");
	// Spawn enemy bot
	CPlayer *enemy_bot = AddBot("Bot" + to_string(GetFirstEmptySlotId()));
	vBots.push_back(enemy_bot);
	vInputInputs.resize(vBots.size());
	enemy_bot->KillCharacter();
	enemy_bot->TryRespawn(right_side ? left_spawn_pos : right_spawn_pos);
	m_pController->m_Teams.SetCharacterTeam(enemy_bot->GetCID(), Team);
	enemy_bot->GetCharacter()->SetActiveWeapon(WEAPON_HAMMER);
	enemy_bot->GetCharacter()->m_TeleCheckpoint = right_side ? 1 : 2;
	enemy_bot->GetCharacter()->Core()->m_Jumps = 0;
	//printf("2\n");

	// Spawn ball
	CPlayer *ball_bot = AddBot("Ball" + to_string(GetFirstEmptySlotId()));
	vBots.push_back(ball_bot);
	ball_bot->KillCharacter();
	ball_bot->TryRespawn(ball_spawn_pos);
	m_pController->m_Teams.SetCharacterTeam(ball_bot->GetCID(), Team);
	ball_bot->GetCharacter()->m_TeleCheckpoint = 3;
	ball_bot->GetCharacter()->Core()->m_Jumps = 0;
	ball_bot->GetCharacter()->SetDeepFrozen(true);
	//printf("3\n");

	m_pController->m_Teams.ResetRoundState(Team);
	m_pController->m_Teams.SetTeamLock(Team, true);
	// Start
	ChangeSwitchState(1, Team, true);
	ChangeSwitchState(2, Team, true);
	ChangeSwitchState(3, Team, true);

	// Ball falling for score mode
	ChangeSwitchState(13, Team, false);
	ChangeSwitchState(10, Team, true);
	ChangeSwitchState(11, Team, true);
	ChangeSwitchState(12, Team, true);
	ChangeSwitchState(14, Team, true);
	ChangeSwitchState(15, Team, true);
	ChangeSwitchState(16, Team, true);
	ChangeSwitchState(20, Team, false);
	ChangeSwitchState(17, Team, true);
	ChangeSwitchState(18, Team, true);
	ChangeSwitchState(19, Team, true);
	ChangeSwitchState(21, Team, true);
	ChangeSwitchState(22, Team, true);
	ChangeSwitchState(23, Team, true);
	ChangeSwitchState(24, Team, true);
	ChangeSwitchState(32, Team, true);

	//printf("4\n");
}

bool CNeuralNetwork::IsTraining()
{
	return is_training;
}

void CNeuralNetwork::OnInit()
{
	m_pServer = (CServer*)Kernel()->RequestInterface<IServer>();
	m_pStorage = Kernel()->RequestInterface<IStorage>();
	m_pConsole = Kernel()->RequestInterface<IConsole>();
	m_pGameContext = (CGameContext *)Kernel()->RequestInterface<IGameServer>();
	m_pCollision = m_pGameContext->Collision();
	m_pController = (CGameControllerDDRace *)m_pGameContext->m_pController;

	unsigned int Seed = 3112; // 3112

	//secure_random_fill(&Seed, sizeof(Seed));

	srand(Seed);

	// Also define NEURAL_NETWORK_TRAINING in Visual Studio settings to speed up ticks and gain more control(disable auto spawn)
	// !!!!!! Change MAX_CLIENTS and NET_MAX_CLIENTS to 64 if not training
	is_training = true;

	skip_tick = 3;
	cache_model_gap = 20;
	count_teams = 42;
	count_bots = count_teams * 3;
	count_player_bots = count_teams * 2;
	available_ticks_to_store = 1024000 / 2; // /2
	count_ticks = available_ticks_to_store / count_player_bots;
	update_tick = count_ticks * skip_tick;
	ticks_collected = last_update_tick = 0;

	const CMapItemLayerTilemap *pTileMap = m_pGameContext->Layers()->GameLayer();
	const CTile *pTiles = static_cast<CTile *>(Kernel()->RequestInterface<IMap>()->GetData(pTileMap->m_Data));

	if(is_training)
	{
		// printf("Creating train directory with folders...\n");
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

		if(fs_makedir(string("train\\" + dir_name + "\\models\\previous").c_str()) != 0)
		{
			dbg_msg("neuralnetwork", "Can't make previous models directory");
			exit(1);
		}

		if(fs_makedir(string("train\\" + dir_name + "\\demos").c_str()) != 0)
		{
			dbg_msg("neuralnetwork", "Can't make demos directory");
			exit(1);
		}
		dbg_msg("neuralnetwork", "Train directory with folders created.");

		dbg_msg("neuralnetwork", "Adding bots...");
		if(count_teams)
		{
			//vBotLastPos.resize(count_player_bots);
			//vBotLastVel.resize(count_player_bots);
			vBallLastPos.resize(count_teams);
			vInputInputs.resize(count_player_bots);
			// vInputBlocks.resize(count_bots);
			vOutputs.resize(count_player_bots);
			// vIsPreviouslyHooked.resize(count_bots);
			// vPrevHookPos.resize(count_bots);
			// vBotsSpawnPos.resize(count_bots);
			// vBotsValidateSpawnPoint.resize(count_bots);
			// vBotsLastCheckPoint.resize(count_bots);
			//vBotsCumulativeRewards.resize(count_player_bots);
			// vBotBestDistance.resize(count_player_bots);

			for(size_t i = 0; i < count_bots; i++)
			{
				std::string name;

				if(i % 3 == 2)
					name = "Ball";
				else
					name = "Bot";

				name += to_string(i);

				auto bot = AddBot(name.c_str(), vec2(32.f, 32.f));
				vBots.push_back(bot);
			}

			for(size_t i = 0; i < count_teams; i++)
			{
				RespawnTeam(i + 1);

				/*if (!model_manager->IsTraining())
				{
					char aFilename[IO_MAX_PATH_LENGTH];
					str_format(aFilename, sizeof(aFilename), "%s_%s_%d_%llu.demo", m_pServer->m_aCurrentMap, name.c_str(), m_pServer->m_NetServer.Address().port, time_get());
					string path_demo = "train/" + dir_name + "/demos/" + aFilename;
					int ret = m_pServer->m_aDemoRecorder[i].Start(m_pStorage, m_pConsole, path_demo.c_str(), m_pGameContext->NetVersion(), m_pServer->m_aCurrentMap, &m_pServer->m_aCurrentMapSha256[CServer::MAP_TYPE_SIX], m_pServer->m_aCurrentMapCrc[CServer::MAP_TYPE_SIX], "server", m_pServer->m_aCurrentMapSize[CServer::MAP_TYPE_SIX], m_pServer->m_apCurrentMapData[CServer::MAP_TYPE_SIX]);
				}*/
			}
		}

		dbg_msg("neuralnetwork", "Bots added");
	}

	dbg_msg("neuralnetwork", "Initializing neural model...");
	model_manager = new ModelManager(is_training, "train\\" + dir_name, available_ticks_to_store, count_player_bots, Seed);
	dbg_msg("neuralnetwork", "Model initialized.");

	if(is_training)
	{
		dbg_msg("neuralnetwork", "Creating data.csv file for statistics...");
		{
			char aFilename[IO_MAX_PATH_LENGTH];
			sprintf_s(aFilename,
				sizeof(aFilename),
				"lr%.1embs%lldppoe%lldbots%drpb%d.csv",
				model_manager->GetLearningRate(),
				model_manager->GetMiniBatchSize(),
				model_manager->GetCountPPOEpochs(),
				count_bots,
				update_tick
			);
			logger.open("train\\" + dir_name + "\\" + aFilename);

			// Define the CSV header using a vector
			std::vector<std::string> header_columns = {
				"Step",
				"Count episodes",
				"Cumulative ball hits",
				"First bot cumulative score",
				"Second bot cumulative score",
				"Average freeze time",
				"Average ball absolute velocity",
				"Average ball velocity(x)",
				"Average first bot reward",
				"Average second bot reward",
				"Highest reward per tick",
				"TPS",
				"Training loss",
				"Actor loss",
				"Critic loss",
				"Critic Mean Absolute Error",
				"Critic Correlation Coefficient",
				"Entropy",
				"Entropy coefficient",
				"Actor grad norm",
				"Critic grad norm",
				"Actor weight norm",
				"Critic weight norm",
				"Actor activation mean",
				"Actor activation std",
				"Learning rate",
				"Time since start",
				"Time to decide",
				"Time to tick",
				"Time rest",
				"Time pre forward",
				"Time forward",
				"Time normal",
				"Time to cpu",
				"Time process last"};

			// Write the CSV header
			for(size_t i = 0; i < header_columns.size(); ++i)
			{
				logger << header_columns[i];
				if(i < header_columns.size() - 1)
				{
					logger << ","; // Add a comma between columns
				}
			}
			logger << endl;
		}
		dbg_msg("neuralnetwork", "data.csv file created and initialized.");
	}

	// Recording demo to spectate how model performs
	string path_demo;
	{
		/*if(model_manager->IsTraining())
		{
			char aFilename[IO_MAX_PATH_LENGTH];
			str_format(aFilename, sizeof(aFilename), "%s_%d_%llu.demo", m_pServer->m_aCurrentMap, m_pServer->m_NetServer.Address().port, time_get());
			path_demo = "train/" + dir_name + "/demos/" + aFilename;
			int ret = m_pServer->m_aDemoRecorder[MAX_CLIENTS].Start(m_pStorage, m_pConsole, path_demo.c_str(), m_pGameContext->NetVersion(), m_pServer->m_aCurrentMap, &m_pServer->m_aCurrentMapSha256[CServer::MAP_TYPE_SIX], m_pServer->m_aCurrentMapCrc[CServer::MAP_TYPE_SIX], "server", m_pServer->m_aCurrentMapSize[CServer::MAP_TYPE_SIX], m_pServer->m_apCurrentMapData[CServer::MAP_TYPE_SIX]);
		}*/
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
	//printf("PreTick\n");
	//auto *switcher = &m_pGameContext->Switchers()[4];
	//// for(auto &Switcher : m_pGameContext->Switchers())
	//for(size_t team_id = 1; team_id <= count_teams; team_id++)
	//{
	//	switcher->m_aStatus[team_id] = true;
	//	switcher->m_aEndTick[team_id] = 0;
	//	switcher->m_aType[team_id] = TILE_SWITCHOPEN;
	//	switcher->m_aLastUpdateTick[team_id] = m_pServer->Tick();
	//}
	//std::cout << (!vBots[1]->GetCharacter()->Switchers().empty() && vBots[1]->GetCharacter()->Team() != TEAM_SUPER && vBots[1]->GetCharacter()->Switchers()[4].m_aStatus[vBots[1]->GetCharacter()->Team()]) << std::endl;
	if((time_get_impl() - ticks_timer) / time_freq() >= 1.0f)
	{
		ticks_per_second = m_pServer->Tick() - start_ticks;
		start_ticks = m_pServer->Tick();
		ticks_timer = time_get_impl();
	}
	auto now = std::chrono::high_resolution_clock::now();
	cumulative_time_rest += std::chrono::duration_cast<std::chrono::duration<float>>(now - decide_time).count() * 1000.f;
	decide_time = now;
	// Handle bots
	if(m_pServer->Tick() % skip_tick == 0 && vBots.size())
	{
		//printf("1111\n");
		gamelayer = m_pGameContext->Layers()->GameLayer();
		pTiles = static_cast<CTile *>(Kernel()->RequestInterface<IMap>()->GetData(gamelayer->m_Data));
		int map_width = gamelayer->m_Width;
		int map_height = gamelayer->m_Height;
		if(respawn_all && IsTraining())
		{
			for(size_t i = 0; i < count_teams; i++)
			{
				RespawnTeam(i + 1);
			}
			respawn_all = false;
		}

		//  apply new input
		// decide_time = time_get_impl();
		int input_counter = 0;
		for(size_t i = 0; i < vBots.size(); i++)
		{
			auto bot = vBots[i];

			auto bot_character = bot->GetCharacter();
			CCharacterCore *bot_character_core;
			// auto bot_2_character = gamecontext->GetPlayerChar(bot_2->GetCID());

			if(!bot_character)
			{
				continue;
			}

			int Team = bot_character->Team();

			// It is ball, because he is deep frozen
			if(bot_character->Core()->m_DeepFrozen)
			{
				if(IsTraining())
					vBallLastPos[Team - 1] = bot_character->m_Pos;
				continue;
			}

			bot_character_core = bot_character->Core();

			int enemy_id = i % 3 == 0 ? i + 1 : i - 1;
			int ball_id = i % 3 == 0 ? i + 2 : i + 1;

			if(!IsTraining() && !GetEnemyAndBallId(bot->GetCID(), Team, enemy_id, ball_id))
			{
				input_counter += 1;
				continue;
			}

			//std::cout << "Enemy id: " << enemy_id << std::endl;
			//std::cout << "Ball id: " << ball_id << std::endl;

			int rotate = bot_character->m_TeleCheckpoint == 1 ? 1 : -1;

			auto enemy_character = m_pGameContext->GetPlayerChar(enemy_id);

			CCharacterCore *enemy_character_core = enemy_character->Core();

			CCharacterCore *ball_character_core = m_pGameContext->GetPlayerChar(ball_id)->Core();

			vec2 bot_pos = bot_character_core->m_Pos;
			vec2 enemy_pos = enemy_character_core->m_Pos;
			vec2 ball_pos = ball_character_core->m_Pos;

			ModelInputInputs *input_inputs = &vInputInputs[input_counter];

			input_inputs->side = -rotate;

			// Local bot
			//printf("12312\n");
			input_inputs->bot_pos = (bot_pos - volleyball_area_center) / (volleyball_area_sizes / 2.f);
			input_inputs->bot_pos.x *= rotate;
			if(input_inputs->bot_pos.x < -1.f || input_inputs->bot_pos.x > 1.f
				|| input_inputs->bot_pos.y < -1.f || input_inputs->bot_pos.y > 1.f)
			{
				input_inputs->bot_pos = {0, 0};
				input_inputs->bot_is_out_of_area = true;
				input_inputs->bot_freeze_time = 1.f;
				input_inputs->bot_hammer_time = 1.f;
			}
			else
			{
				input_inputs->bot_is_out_of_area = false;
				input_inputs->bot_freeze_time = std::clamp(bot_character->m_FreezeTime / (3.f * (float)SERVER_TICK_SPEED), 0.f, 1.f);
				input_inputs->bot_hammer_time = (float)bot_character->m_ReloadTimer / 16.f;
			}
			input_inputs->bot_vel = bot_character_core->m_Vel / 40.f;
			input_inputs->bot_vel.x *= rotate;

			input_inputs->bot_is_hooking = bot_character_core->m_HookState == HOOK_FLYING || bot_character_core->m_HookState == HOOK_GRABBED;
			input_inputs->bot_is_grabbed = bot_character_core->m_HookState == HOOK_GRABBED;
			input_inputs->bot_is_retracted = bot_character_core->m_HookState == HOOK_RETRACTED || (bot_character_core->m_HookState >= HOOK_RETRACT_START && bot_character_core->m_HookState <= HOOK_RETRACT_END);
			input_inputs->bot_hook_time = 1.f - bot_character_core->m_HookTick / ((float)SERVER_TICK_SPEED * 1.25f);

			if(!input_inputs->bot_is_out_of_area && input_inputs->bot_is_hooking)
			{
				auto hook_relative = (bot_character_core->m_HookPos - bot_character_core->m_Pos) / m_pGameContext->Tuning()->m_HookLength;

				input_inputs->bot_hook_pos = (bot_character_core->m_HookPos - volleyball_area_center) / (volleyball_area_sizes / 2.f);
				input_inputs->bot_hook_pos.x *= rotate;
				input_inputs->bot_hook_dir = bot_character_core->m_HookDir;
				input_inputs->bot_hook_dir.x *= rotate;

				auto ataned = atan2(hook_relative.y, hook_relative.x);
				auto angle_x = cos(ataned);
				auto angle_y = sin(ataned);
				input_inputs->bot_hook_angle = vec2(angle_x, angle_y);
				input_inputs->bot_hook_angle.x *= rotate;
			}
			else
			{
				input_inputs->bot_hook_pos = input_inputs->bot_hook_dir = input_inputs->bot_hook_angle = vec2(0, 0);
			}

			// Enemy

			input_inputs->enemy_pos = (enemy_pos - volleyball_area_center) / (volleyball_area_sizes / 2.f);
			input_inputs->enemy_pos.x *= rotate;
			if(input_inputs->enemy_pos.x < -1.f || input_inputs->enemy_pos.x > 1.f || input_inputs->enemy_pos.y < -1.f || input_inputs->enemy_pos.y > 1.f)
			{
				input_inputs->enemy_pos = {0, 0};
				input_inputs->enemy_is_out_of_area = true;
				input_inputs->enemy_freeze_time = 1.f;
				input_inputs->enemy_hammer_time = 1.f;
			}
			else
			{
				input_inputs->enemy_is_out_of_area = false;
				input_inputs->enemy_freeze_time = std::clamp(enemy_character->m_FreezeTime / (3.f * (float)SERVER_TICK_SPEED), 0.f, 1.f);
				input_inputs->enemy_hammer_time = (float)enemy_character->m_ReloadTimer / 16.f;
			}
			input_inputs->enemy_vel = enemy_character_core->m_Vel / 40.f;
			input_inputs->enemy_vel.x *= rotate;

			input_inputs->enemy_is_hooking = enemy_character_core->m_HookState == HOOK_FLYING || enemy_character_core->m_HookState == HOOK_GRABBED;
			input_inputs->enemy_is_grabbed = enemy_character_core->m_HookState == HOOK_GRABBED;
			input_inputs->enemy_is_retracted = enemy_character_core->m_HookState == HOOK_RETRACTED || (enemy_character_core->m_HookState >= HOOK_RETRACT_START && enemy_character_core->m_HookState <= HOOK_RETRACT_END);
			input_inputs->enemy_hook_time = 1.f - enemy_character_core->m_HookTick / ((float)SERVER_TICK_SPEED * 1.25f);

			if(!input_inputs->enemy_is_out_of_area && input_inputs->enemy_is_hooking)
			{
				auto hook_relative = (enemy_character_core->m_HookPos - enemy_character_core->m_Pos) / m_pGameContext->Tuning()->m_HookLength;

				input_inputs->enemy_hook_pos = (enemy_character_core->m_HookPos - volleyball_area_center) / (volleyball_area_sizes / 2.f);
				input_inputs->enemy_hook_pos.x *= rotate;
				input_inputs->enemy_hook_dir = enemy_character_core->m_HookDir;
				input_inputs->enemy_hook_dir.x *= rotate;

				auto ataned = atan2(hook_relative.y, hook_relative.x);
				auto angle_x = cos(ataned);
				auto angle_y = sin(ataned);
				input_inputs->enemy_hook_angle = vec2(angle_x, angle_y);
				input_inputs->enemy_hook_angle.x *= rotate;
			}
			else
			{
				input_inputs->enemy_hook_pos = input_inputs->enemy_hook_dir = input_inputs->enemy_hook_angle = vec2(0, 0);
			}

			//
			// Ball
			//

			input_inputs->ball_pos = (ball_pos - volleyball_area_center) / (volleyball_area_sizes / 2.f);
			input_inputs->ball_pos.x *= rotate;
			
			if(input_inputs->ball_pos.x < -1.f || input_inputs->ball_pos.x > 1.f || input_inputs->ball_pos.y < -1.f || input_inputs->ball_pos.y > 1.f)
			{
				input_inputs->ball_pos = {0, 0};
				input_inputs->ball_is_out_of_area = true;
			}
			else
				input_inputs->ball_is_out_of_area = false;

			input_inputs->ball_vel = ball_character_core->m_Vel / 40.f;
			input_inputs->ball_vel.x *= rotate;
			//vBotLastPos[i - ((i + 1) / 3)] = bot_pos;
			//vBotLastVel[i - ((i + 1) / 3)] = length(bot_character_core->m_Vel);

			input_counter += 1;
		}
		// cout << "Time calcs: " << (float)summerr / (float)time_freq() << endl;

		auto now = std::chrono::high_resolution_clock::now();
		cumulative_time_rest += std::chrono::duration_cast<std::chrono::duration<float>>(now - decide_time).count() * 1000.f;
		// cout << "Time rest: " << std::chrono::duration_cast<std::chrono::duration<float>>(now - decide_time).count() << endl;
		double time_pre_forward = 0;
		double time_forward = 0;
		double time_normal = 0;
		double time_to_cpu = 0;
		double time_process_last = 0;
		decide_time = std::chrono::high_resolution_clock::now();
		//printf("pree\n");
		try
		{
			vOutputs = model_manager->Decide(vInputInputs, time_pre_forward, time_forward, time_normal, time_to_cpu, time_process_last);
		}
		catch(const std::exception &e)
		{
			std::cout << "Decide ended with an error: " << e.what() << std::endl;
			system("PAUSE");
		}
		//printf("postee\n");

		now = std::chrono::high_resolution_clock::now();
		//auto dur = std::chrono::duration_cast<std::chrono::duration<float>>(now - decide_time).count() * 1000.f;
		//std::cout << dur << std::endl;
		/*if(dur > 30.f)
		{
			dbg_msg("neuralnetwork", "Time: %f", dur);
		}*/
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
	//printf("Donee\n");
}

void CNeuralNetwork::PreOnClientPredictedEarlyInput()
{
	//for(size_t i = 0; i < vBots.size(); i++)
	//{
	//	auto bot = vBots[i];

	//	bot->UpdatePlaytime();

	//	if(i % 3 == 2)
	//	{
	//		continue;
	//	}

	//	//  Move bot wherever you want
	//	// m_Direction:
	//	// 1 - Right
	//	// 0 - Stay
	//	// -1 - Left

	//	int angle = 0;

	//	int rotate = i % 3 == 0 ? 1 : -1;

	//	auto bot_character = bot->GetCharacter();
	//	int model_jump = 0;
	//	vec2 model_angle;
	//	int model_direction = 0;
	//	int model_hook = 0;
	//	int model_hammer = 0;

	//	if(bot_character != nullptr)
	//	{
	//		ModelOutput returned_model = vOutputs[i - ((i + 1) / 3)]; // {0.0f}; //
	//		model_angle = returned_model.angle * 299.f;
	//		model_angle.x *= rotate;
	//		model_direction = returned_model.direction * rotate;
	//		model_hook = returned_model.hook;
	//		model_hammer = returned_model.hammer;
	//	}

	//	CNetObj_PlayerInput pApplyInput;
	//	mem_zero(&pApplyInput, sizeof(pApplyInput));
	//	pApplyInput.m_TargetX = (int)round(model_angle.x);
	//	pApplyInput.m_TargetY = (int)round(model_angle.y);
	//	// pApplyInput.m_Jump = model_jump;
	//	pApplyInput.m_Direction = model_direction;
	//	pApplyInput.m_Hook = model_hook;
	//	pApplyInput.m_WantedWeapon = WEAPON_HAMMER;
	//	pApplyInput.m_Fire = model_hammer;

	//	m_pGameContext->OnClientPredictedEarlyInput(bot->GetCID(), &pApplyInput);
	//}
}

void CNeuralNetwork::PreOnClientPredictedInput()
{
	//printf("PreOnClientPredictedInput\n");
	int output_counter = 0;
	for(size_t i = 0; i < vBots.size() && vOutputs.size(); i++)
	{
		auto bot = vBots[i];

		bot->UpdatePlaytime();

		if(!bot->GetCharacter() || bot->GetCharacter()->Core()->m_DeepFrozen)
		{
			continue;
		}

		//  Move bot wherever you want
		// m_Direction:
		// 1 - Right
		// 0 - Stay
		// -1 - Left

		auto bot_character = bot->GetCharacter();

		int rotate = bot_character->m_TeleCheckpoint == 1 ? 1 : -1;

		int model_jump = 0;
		vec2 model_angle;
		int model_direction = 0;
		int model_hook = 0;
		int model_hammer = 0;

		if(bot_character != nullptr)
		{
			ModelOutput returned_model = vOutputs[output_counter]; // {0.0f}; //
			model_angle = returned_model.angle * 299.f;
			model_angle.x *= rotate;
			model_direction = returned_model.direction * rotate;
			model_hook = returned_model.hook;
			model_hammer = returned_model.hammer;
		}

		CNetObj_PlayerInput pApplyInput;
		mem_zero(&pApplyInput, sizeof(pApplyInput));
		pApplyInput.m_TargetX = (int)round(model_angle.x);
		pApplyInput.m_TargetY = (int)round(model_angle.y);
		//pApplyInput.m_Jump = model_jump;
		pApplyInput.m_Direction = model_direction;
		pApplyInput.m_Hook = model_hook;
		pApplyInput.m_Fire = model_hammer;

		m_pGameContext->OnClientPredictedEarlyInput(bot->GetCID(), &pApplyInput);
		output_counter += 1;
	}
}

void CNeuralNetwork::PostTick(float time_to_tick)
{
	//printf("PostTick\n");
	static float freeze_cumulative_time = 0;
	static float first_bot_cumulative_reward = 0;
	static float second_bot_cumulative_reward = 0;
	static float first_bot_cumulative_score = 0;
	static float second_bot_cumulative_score = 0;
	static float cumulative_ball_speed_x = 0;
	static float cumulative_ball_abs_speed = 0;
	static float cumulative_ball_hits = 0;
	static float highest_reward_per_tick = 0;
	static int count_updated = 0;

	// Rewards
	static float goal_reward = 5.f; // Rewards when scoaring a goal
	static float goal_penalize_reward = -3.f; // Penalizes if goaled on your side

	static float ball_on_spawn_reward = -0.1f; // -1.f There are 3 spawns. Center(at the start), left side and right side
	static float ball_on_side_reward = 0.1f; // 0.05f If the ball is on your side it penalizes you, otherwise rewards you
	static float ball_on_side_distance_reward = 0.1f; // 0.2f It means that if the ball is on your side and far from the net it always penalize you on that reward, if ball is half closer to net it penalize on half, but if on enemy side it rewards
	static float ball_moving_towards_net_reward = 0.3f; // 0.2f 10 If the ball is on your side it rewards for moving towards net, otherwise penalize. For example if ball moves from the farthest point to net in summ it would be this reward, so it calculates delta of moving to the net in %
	static float ball_moving_towards_goal_reward = 0.03f; // 0.2f 10 On the enemy side it rewards if ball is moving toward goal
	static float being_in_freeze_reward = -0.2f; // -0.1f if the bot is currently freezed it penalizes you on that reward
	static float bot_is_grabbed_reward = 0.1f; // If the bot is currently grabbed to wall/ball applies to every tick
	static float bot_is_holding_ball_reward = 0.05f; // If the bot is currently holding ball using hook it rewards every tick
	static float bot_moving_towards_ball_reward = 0.1f; // Not implemented
	static float bot_hitted_ball_reward = 2.0f; // Rewards bot for hitting ball
	static float bot_hook_missed_reward = -0.5f; // Applies when bot teleported with ground teleporter
	static float bot_teleported_reward = -1.f; // Applies when bot teleported with ground teleporter
	static float step_reward = -0.02f; // -0.001f Applies every tick
	static float divide_reward_by = 5.f;

	static int long_no_improvements_ticks = 200;

	//static std::vector<float> rewards;
	static float best_average = 999999999999.f;
	static float last_saved = 9999999999999.f;

	auto now = std::chrono::high_resolution_clock::now();
	cumulative_time_to_tick += time_to_tick * 1000.f;

	// size_t summerr = 0;
	auto measure_rest = std::chrono::high_resolution_clock::now();
	if(m_pServer->Tick() % skip_tick == 0)
	{
		for(size_t team_id = 0; team_id < count_teams && IsTraining(); team_id++)
		{
			bool match_is_done = false;

			int first_bot_id = team_id * 3;
			int second_bot_id = team_id * 3 + 1;
			int ball_id = team_id * 3 + 2;

			auto first_bot_character = vBots[first_bot_id]->GetCharacter();
			auto second_bot_character = vBots[second_bot_id]->GetCharacter();
			auto ball_character = vBots[ball_id]->GetCharacter();

			int teleport_num = ball_character->m_TeleportNum;
			ball_character->m_TeleportNum = 0;

			auto first_bot_pos = first_bot_character->m_Pos;
			auto second_bot_pos = second_bot_character->m_Pos;
			auto ball_pos = ball_character->m_Pos;

			float reward = 0;
			float first_bot_reward = step_reward;
			float second_bot_reward = step_reward;

			bool goaled = false;

			if(teleport_num == 1 || teleport_num == 2)
			{
				//reward += teleport_num == 1 ? -goal_reward : goal_reward;
				//std::cout << teleport_num << std::endl;
				if (teleport_num == 1)
				{
					first_bot_reward += goal_penalize_reward;
					second_bot_reward += goal_reward;
					second_bot_cumulative_score += 1;
				}
				else
				{
					first_bot_reward += goal_reward;
					second_bot_reward += goal_penalize_reward;
					first_bot_cumulative_score += 1;
				}
				goaled = true;
			}

			if(first_bot_character->m_TeleportNum == 1)
			{
				first_bot_character->m_TeleportNum = 0;
				first_bot_reward += bot_teleported_reward;
			}
			
			if(second_bot_character->m_TeleportNum == 2)
			{
				second_bot_character->m_TeleportNum = 0;
				second_bot_reward += bot_teleported_reward;
			}

			first_bot_reward += first_bot_character->GetCore().m_HookState == HOOK_GRABBED ? bot_is_grabbed_reward : 0;
			second_bot_reward += second_bot_character->GetCore().m_HookState == HOOK_GRABBED ? bot_is_grabbed_reward : 0;

			// Handle miss of the hook
			if(first_bot_character->m_HookMissed)
			{
				first_bot_character->m_HookMissed = false;
				first_bot_reward += bot_hook_missed_reward;
			}

			if(second_bot_character->m_HookMissed)
			{
				second_bot_character->m_HookMissed = false;
				second_bot_reward += bot_hook_missed_reward;
			}

			// Handle ball hit
			if (first_bot_character->m_HittedBall)
			{
				first_bot_character->m_HittedBall = false;
				first_bot_reward += bot_hitted_ball_reward;
				cumulative_ball_hits += 1;
			}

			if(second_bot_character->m_HittedBall)
			{
				second_bot_character->m_HittedBall = false;
				second_bot_reward += bot_hitted_ball_reward;
				cumulative_ball_hits += 1;
			}

			if (first_bot_character->GetCore().m_HookedPlayer && first_bot_character->GetCore().m_HookedPlayer % 3 == 2)
				first_bot_reward += bot_is_holding_ball_reward;

			if(second_bot_character->GetCore().m_HookedPlayer && second_bot_character->GetCore().m_HookedPlayer % 3 == 2)
				second_bot_reward += bot_is_holding_ball_reward;

			first_bot_reward += first_bot_character->m_FreezeTime ? being_in_freeze_reward : 0;
			second_bot_reward += second_bot_character->m_FreezeTime ? being_in_freeze_reward : 0;

			if(IsSwitchEnabled(4, team_id+1))
			{
				match_is_done = true;

				// Respawn everyone in team
				RespawnTeam(team_id + 1);
			}
			else
			{
				if(ball_pos.x > volleyball_area_start.x && ball_pos.y > volleyball_area_start.y && ball_pos.x < volleyball_area_end.x && ball_pos.y < volleyball_area_end.y)
				{
					cumulative_ball_speed_x += ball_character->Core()->m_Vel.x;
					cumulative_ball_abs_speed += length(ball_character->Core()->m_Vel);
					freeze_cumulative_time += first_bot_character->m_FreezeTime ? 1 : 0;
					freeze_cumulative_time += second_bot_character->m_FreezeTime ? 1 : 0;

					float ball_to_line_distance_normalized = ClosestDistanceToDividingLine(ball_pos) / sqrtf(pow(13.5f, 2) + pow(9, 2)) / 32.f;
					if(ball_to_line_distance_normalized > 1.f)
					{
						std::cout << ball_to_line_distance_normalized << std::endl;
					}

					vec2 ball_last_pos = vBallLastPos[team_id];
					if(ball_last_pos.x > volleyball_area_start.x \
						&& ball_last_pos.y > volleyball_area_start.y \
						&& ball_last_pos.x < volleyball_area_end.x \
						&& ball_last_pos.y < volleyball_area_end.y)
					{
						float ball_to_line_last_distance_normalized = ClosestDistanceToDividingLine(ball_last_pos) / sqrtf(pow(13.5f, 2) + pow(9, 2)) / 32.f;
						float ball_to_goal_distance_normalized = (abs(ball_last_pos.y - 118.5f * 32.f) - abs(ball_pos.y - 118.5f * 32.f)) / 32.f / 20.f;
						float distance_change_reward = (ball_to_line_last_distance_normalized - ball_to_line_distance_normalized) * ball_moving_towards_net_reward;
						float distance_to_goal_reward = ball_to_goal_distance_normalized * ball_moving_towards_goal_reward;

						if(ball_pos.x < volleyball_area_center.x)
						{
							first_bot_reward += distance_change_reward;
							second_bot_reward += distance_to_goal_reward;
						}
						else if(ball_pos.x > volleyball_area_center.x)
						{
							first_bot_reward += distance_to_goal_reward;
							second_bot_reward += distance_change_reward;
						}
					}
					//std::cout << ball_to_line_distance_normalized << std::endl;
					//std::cout << ball_pos.x / 32.f << " " << ball_pos.y / 32.f << std::endl;
					int cur_block_x = (int)(ball_pos.x / 32.f);
					int cur_block_y = (int)(ball_pos.y / 32.f);

					if(volleyball_ball_center_spawn.x == cur_block_x && volleyball_ball_center_spawn.y == cur_block_y)
					{
						first_bot_reward += ball_on_spawn_reward;
						second_bot_reward += ball_on_spawn_reward;
						//printf("On center spawn\n");
					}
			
					if(ball_pos.x < volleyball_area_center.x)
					{
						reward -= ball_on_side_reward + ball_on_side_distance_reward * ball_to_line_distance_normalized;
						if(volleyball_ball_left_side_spawn.x == cur_block_x && volleyball_ball_left_side_spawn.y == cur_block_y)
						{
							first_bot_reward += ball_on_spawn_reward;
							//printf("On spawn left\n");
						}
					}
					else if (ball_pos.x > volleyball_area_center.x)
					{
						reward += ball_on_side_reward + ball_on_side_distance_reward * ball_to_line_distance_normalized;
						if(volleyball_ball_right_side_spawn.x == cur_block_x && volleyball_ball_right_side_spawn.y == cur_block_y)
						{
							second_bot_reward += ball_on_spawn_reward;
							//printf("On spawn right\n");
						}
					}
				}
			}

			first_bot_reward += reward;
			second_bot_reward -= reward;

			if(first_bot_reward > highest_reward_per_tick)
				highest_reward_per_tick = first_bot_reward;
			if(second_bot_reward > highest_reward_per_tick)
				highest_reward_per_tick = second_bot_reward;

			first_bot_cumulative_reward += first_bot_reward;
			second_bot_cumulative_reward += second_bot_reward;
			model_manager->Reward(first_bot_reward / divide_reward_by, false, match_is_done);
			model_manager->Reward(second_bot_reward / divide_reward_by, false, match_is_done);
		}
		bool is_full = false;

		model_manager->SaveReplays(is_full);

		auto now = std::chrono::high_resolution_clock::now();
		cumulative_time_rest += std::chrono::duration_cast<std::chrono::duration<float>>(now - measure_rest).count() * 1000.f;

		ticks_collected += 1;

		if(is_full && IsTraining())
		{
			//printf("Updating\n");
			// decide_time = time_get_impl();

			float avg_first_bot_reward = first_bot_cumulative_reward / (float)(count_teams);
			float avg_second_bot_reward = second_bot_cumulative_reward / (float)(count_teams);
			float avg_freeze_time = freeze_cumulative_time / ((float)(m_pServer->Tick() - last_update_tick) / (float)skip_tick) / (float)count_player_bots;
			float avg_ball_abs_vel = cumulative_ball_abs_speed / ((float)(m_pServer->Tick() - last_update_tick) / (float)skip_tick);
			float avg_ball_vel_x = cumulative_ball_speed_x / ((float)(m_pServer->Tick() - last_update_tick) / (float)skip_tick);

			auto demo_recorder = &m_pServer->m_aDemoRecorder[MAX_CLIENTS];

			if(demo_recorder->IsRecording() && IsTraining())
			{
				demo_recorder->Stop();
				char aNewFilename[IO_MAX_PATH_LENGTH];
				str_format(aNewFilename, sizeof(aNewFilename), "average_freeze_%.2f_%s_%llu.demo", avg_freeze_time, m_pServer->m_aCurrentMap, time_get_impl());
				string path_demo = "train/" + dir_name + "/demos/" + aNewFilename;
				m_pStorage->RenameFile(demo_recorder->GetCurrentFilename(), path_demo.c_str(), IStorage::TYPE_ABSOLUTE);
			}
		
			// int64_t update_time = time_get_impl();
			double avg_training_loss = 0;
			double avg_actor_loss = 0;
			double avg_critic_loss = 0;
			double avg_entropy = 0;
			double avg_actor_grad_norm = 0, avg_critic_grad_norm = 0,
			avg_actor_weight_norm = 0, avg_critic_weight_norm = 0,
			avg_actor_activation_mean = 0, avg_actor_activation_std = 0,
			       critic_mean_absolute_error = 0, critic_correlation_coefficient = 0;
			bool updated = false;
			size_t count_episodes = model_manager->GetCountEpisodes();

			float avg_first_bot_score = first_bot_cumulative_score / ((float)(count_episodes) / 2.f);
			float avg_second_bot_score = second_bot_cumulative_score / ((float)(count_episodes) / 2.f);
			float avg_ball_hits = cumulative_ball_hits / ((float)(count_episodes) / 2.f);

			static size_t count_episodes_processed = 0;
			static size_t count_every_update = 0;
			bool cache_model = count_updated % cache_model_gap == 0 && model_manager->IsTraining();
			model_manager->Update(avg_freeze_time, cache_model, updated,
				avg_training_loss, avg_actor_loss, avg_critic_loss,
				avg_entropy,
				avg_actor_grad_norm, avg_critic_grad_norm,
				avg_actor_weight_norm, avg_critic_weight_norm,
				avg_actor_activation_mean, avg_actor_activation_std, critic_mean_absolute_error, critic_correlation_coefficient);
			count_episodes_processed += count_episodes;
			count_every_update += 1;
			auto update_tick_delta = m_pServer->Tick() - last_update_tick;
			last_update_tick = m_pServer->Tick();
			// cout << "Time update: " << (float)(time_get_impl() - decide_time) / (float)time_freq() << endl;
			if(updated)
			{
				count_updated += 1;
				//rewards.clear();
				if(last_saved - avg_freeze_time > last_saved * 0.1f && last_saved - avg_freeze_time > 25)
				{
					if(!m_pStorage->CpyFile(("train\\" + dir_name + "\\models\\last_model.pt").c_str(), ("train\\" + dir_name + "\\models\\early_stopping_" + to_string(avg_freeze_time) + "_model.pt").c_str(), false))
					{
						dbg_msg("neuralnetwork", "Failed to copy early stopped model");
					}
					if(!m_pStorage->CpyFile(("train\\" + dir_name + "\\models\\last_optimizer.pt").c_str(), ("train\\" + dir_name + "\\models\\early_stopping_" + to_string(avg_freeze_time) + "_optimizer.pt").c_str(), false))
					{
						dbg_msg("neuralnetwork", "Failed to copy early stopped optimizer");
					}
					//model_manager->Save("train\\" + dir_name + "\\models\\early_stopping_" + to_string(avg_dist));
					last_saved = avg_freeze_time;
				}

				if(avg_freeze_time < best_average && model_manager->IsTraining())
				{
					best_average = avg_freeze_time;
					if(!m_pStorage->CpyFile(("train\\" + dir_name + "\\models\\last_model.pt").c_str(), ("train\\" + dir_name + "\\models\\best_model.pt").c_str(), false))
					{
						dbg_msg("neuralnetwork", "Failed to copy best model");
					}
					if(!m_pStorage->CpyFile(("train\\" + dir_name + "\\models\\last_optimizer.pt").c_str(), ("train\\" + dir_name + "\\models\\best_optimizer.pt").c_str(), false))
					{
						dbg_msg("neuralnetwork", "Failed to copy best optimizer");
					}
				}

				model_manager->Save("train\\" + dir_name + "\\models\\last");

				logger << ticks_collected / count_ticks
				       << "," << count_episodes
				       << "," << cumulative_ball_hits
				       << "," << first_bot_cumulative_score
				       << "," << second_bot_cumulative_score
				       << "," << avg_freeze_time
				       << "," << avg_ball_abs_vel
				       << "," << avg_ball_vel_x
				       << "," << avg_first_bot_reward
				       << "," << avg_second_bot_reward
				       << "," << highest_reward_per_tick
				       << "," << ticks_per_second
				       << "," << avg_training_loss
				       << "," << avg_actor_loss
				       << "," << avg_critic_loss
				       << "," << critic_mean_absolute_error
				       << "," << critic_correlation_coefficient
				       << "," << avg_entropy
				       << "," << model_manager->GetEntropyCoefficient()
				       << "," << avg_actor_grad_norm
				       << "," << avg_critic_grad_norm
				       << "," << avg_actor_weight_norm
				       << "," << avg_critic_weight_norm
				       << "," << avg_actor_activation_mean
				       << "," << avg_actor_activation_std
				       << "," << model_manager->GetCurrentLearningRate()
				       << "," << (float)time_get_impl() / (float)time_freq()
				       << "," << (cumulative_time_to_decide / (float)update_tick_delta)
				       << "," << (cumulative_time_to_tick / (float)update_tick_delta)
				       << "," << (cumulative_time_rest / (float)update_tick_delta)
				       << "," << cumulative_time_pre_forward / (float)update_tick_delta
				       << "," << cumulative_time_forward / (float)update_tick_delta
				       << "," << cumulative_time_normal / (float)update_tick_delta
				       << "," << cumulative_time_to_cpu / (float)update_tick_delta
				       << "," << cumulative_time_process_last / (float)update_tick_delta
				       << endl;
				dbg_msg("neuralnetwork",\
					"Avg. first/second bot score: %f/%f "\
					"Avg. freeze time: %.2f "\
					"Avg ball abs vel: %.2f "\
					" Avg ball vel(x) : % f "\
					"Avg first/second bot reward: %.2f/%.2f "\
					"Avg/Cumulative ball hits: %.2f/%.0f "\
					"TPS: %d "\
					"Avg. Training/Actor/Critic Loss: %f/%f/%f "\
					"Avg. Entropy: %f "\
					"Episodes: %d/%d "\
					"Updates: %d", \
					avg_first_bot_score, avg_second_bot_score, \
					avg_freeze_time, \
					avg_ball_abs_vel, \
					avg_ball_vel_x, \
					avg_first_bot_reward, avg_second_bot_reward, \
					avg_ball_hits, cumulative_ball_hits, \
					ticks_per_second, \
					avg_training_loss, avg_actor_loss, avg_critic_loss, \
					avg_entropy, \
					count_episodes, count_episodes_processed, \
					count_every_update);
					
				/*cout << "Avg. reward: " << avg_reward << " TPS: " << ticks_per_second << " Avg. Training Loss: " << avg_training_loss
					    << " Dies: " << dies << " Episodes: " << count_episodes << "/" << count_episodes_processed << " Updates: " << count_every_update
					    << " Avg. distance: " << avg_dist << " Avg. valid distance: " << avg_valid_dist << endl;*/
				first_bot_cumulative_score \
				= second_bot_cumulative_score \
				= first_bot_cumulative_reward \
				= second_bot_cumulative_reward \
				= highest_reward_per_tick \
				= cumulative_ball_hits \
				= freeze_cumulative_time \
				= cumulative_ball_speed_x \
				= cumulative_ball_abs_speed \
				= cumulative_time_to_decide \
				= cumulative_time_to_tick \
				= cumulative_time_rest \
				= cumulative_time_pre_forward \
				= cumulative_time_forward \
				= cumulative_time_normal \
				= cumulative_time_normal \
				= cumulative_time_to_cpu \
				= cumulative_time_process_last \
				= count_episodes_processed \
				= count_every_update = 0;
			}
			respawn_all = true;
			// cout << "Time to update: " << (float)(time_get_impl() - update_time) / (float)time_freq() << endl;
			//printf("end\n");
			/*if(updated && count_updated % 20 == 1 && model_manager->IsTraining())
			{
				char aFilename[IO_MAX_PATH_LENGTH];
				str_format(aFilename, sizeof(aFilename), "%s_%d_%llu.demo", m_pServer->m_aCurrentMap, m_pServer->m_NetServer.Address().port, time_get());
				string path_demo = "train\\" + dir_name + "\\demos\\" + aFilename;
				int ret = demo_recorder->Start(m_pStorage, m_pConsole, path_demo.c_str(), m_pGameContext->NetVersion(), m_pServer->m_aCurrentMap, &m_pServer->m_aCurrentMapSha256[CServer::MAP_TYPE_SIX], m_pServer->m_aCurrentMapCrc[CServer::MAP_TYPE_SIX], "server", m_pServer->m_aCurrentMapSize[CServer::MAP_TYPE_SIX], m_pServer->m_apCurrentMapData[CServer::MAP_TYPE_SIX]);
			}*/
			decide_time = std::chrono::high_resolution_clock::now();
			//printf("Updating done\n");
		}
	}

	//printf("21\n");
	/*auto now = std::chrono::high_resolution_clock::now();
	cumulative_time_to_tick += std::chrono::duration_cast<std::chrono::duration<float>>(now - decide_time).count() * 1000.f;*/
	//cout << "Time to tick: " << (float)(now - decide_time) / (float)time_freq() << endl;
	decide_time = std::chrono::high_resolution_clock::now();
}