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
CGameControllerDDRace *m_pController;

vec2 first_bot_spawn_pos = {28.5 * 32.f, 20.5 * 32.f};
vec2 second_bot_spawn_pos = {34.5 * 32.f, 20.5 * 32.f};
vec2 ball_spawn_pos = {24.5 * 32.f, 24.5 * 32.f};

vec2 volleyball_area_start = {160 * 32.f, 98 * 32.f};
vec2 volleyball_area_end = {187 * 32.f, 119 * 32.f};
vec2 volleyball_area_sizes = volleyball_area_end - volleyball_area_start;
vec2 volleyball_area_center = (volleyball_area_start + volleyball_area_end) / 2.f;
vec2 volleyball_area_left_side = (volleyball_area_start + volleyball_area_end) / 2.f;
vec2 volleyball_area_right_side = (volleyball_area_start + volleyball_area_end) / 2.f;

vec2 volleyball_final_area_right_top_corner = {67 * 32.f, 60 * 32.f};

CPlayer *CNeuralNetwork::AddBot(const char *name, vec2 spawn_pos)
{
	for(int ClientID = MAX_CLIENTS - 1; ClientID >= 0; ClientID--)
	{
		if(m_pServer->m_aClients[ClientID].m_State == CServer::CClient::STATE_EMPTY)
		{
			// m_aClients[ClientID].m_aName = "Bot";
			memcpy(m_pServer->m_aClients[ClientID].m_aName, name, strlen(name) + 1);

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

void CNeuralNetwork::OnInit()
{
	m_pServer = (CServer*)Kernel()->RequestInterface<IServer>();
	m_pStorage = Kernel()->RequestInterface<IStorage>();
	m_pConsole = Kernel()->RequestInterface<IConsole>();
	m_pGameContext = (CGameContext *)Kernel()->RequestInterface<IGameServer>();
	m_pController = (CGameControllerDDRace *)m_pGameContext->m_pController;

	unsigned int Seed = 3112; // 3112

	//secure_random_fill(&Seed, sizeof(Seed));

	srand(Seed);

	//gen = mt19937(rd());
	gen = mt19937(Seed);

	spawn_probabilities_updated = false;

	skip_tick = 3;
	count_teams = 42;
	count_bots = count_teams * 3;
	count_player_bots = count_teams * 2;
	available_ticks_to_store = 1024000;
	count_ticks = available_ticks_to_store / count_player_bots;
	update_tick = count_ticks * skip_tick;
	ticks_collected = last_update_tick = 0;

	const CMapItemLayerTilemap *pTileMap = m_pGameContext->Layers()->GameLayer();
	const CTile *pTiles = static_cast<CTile *>(Kernel()->RequestInterface<IMap>()->GetData(pTileMap->m_Data));

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
	dbg_msg("neuralnetwork", "Train directory with folders created.");

	dbg_msg("neuralnetwork", "Adding bots...");
	if(count_teams)
	{
		vBotLastPos.resize(count_player_bots);
		vBotLastVel.resize(count_player_bots);
		vInputInputs.resize(count_player_bots);
		//vInputBlocks.resize(count_bots);
		vOutputs.resize(count_player_bots);
		//vIsPreviouslyHooked.resize(count_bots);
		//vPrevHookPos.resize(count_bots);
		//vBotsSpawnPos.resize(count_bots);
		//vBotsValidateSpawnPoint.resize(count_bots);
		//vBotsLastCheckPoint.resize(count_bots);
		vBotsCumulativeRewards.resize(count_player_bots);
		//vBotBestDistance.resize(count_player_bots);

		for(size_t i = 0; i < count_bots; i++)
		{
			bool is_ball = false;
			int team_id = i / 3 + 1;

			std::string name;
			vec2 spawn_pos;

			if (i % 3 == 2)
			{
				name = "Ball";
				spawn_pos = ball_spawn_pos;
				is_ball = true;
			}
			else
			{
				name = "Bot";
				spawn_pos = i % 3 == 0 ? first_bot_spawn_pos : second_bot_spawn_pos;
			}

			name += to_string(i);

			auto bot = AddBot(name.c_str(), spawn_pos);
			m_pController->m_Teams.SetForceCharacterTeam(bot->GetCID(), team_id);
			bot->GetCharacter()->m_TeleCheckpoint = i % 3 + 1;
			if(is_ball)
			{
				m_pController->m_Teams.SetTeamLock(team_id, true);
			}

			vBots.push_back(bot);

			if (!model_manager->IsTraining())
			{
				char aFilename[IO_MAX_PATH_LENGTH];
				str_format(aFilename, sizeof(aFilename), "%s_%s_%d_%llu.demo", m_pServer->m_aCurrentMap, name.c_str(), m_pServer->m_NetServer.Address().port, time_get());
				string path_demo = "train/" + dir_name + "/demos/" + aFilename;
				int ret = m_pServer->m_aDemoRecorder[i].Start(m_pStorage, m_pConsole, path_demo.c_str(), m_pGameContext->NetVersion(), m_pServer->m_aCurrentMap, &m_pServer->m_aCurrentMapSha256[CServer::MAP_TYPE_SIX], m_pServer->m_aCurrentMapCrc[CServer::MAP_TYPE_SIX], "server", m_pServer->m_aCurrentMapSize[CServer::MAP_TYPE_SIX], m_pServer->m_apCurrentMapData[CServer::MAP_TYPE_SIX]);
			}
		}
	}

	dbg_msg("neuralnetwork", "Bots added");

	dbg_msg("neuralnetwork", "Initializing neural model...");
	model_manager = new ModelManager(count_player_bots * update_tick / skip_tick, count_player_bots, Seed);
	dbg_msg("neuralnetwork", "Model initialized.");

	dbg_msg("neuralnetwork", "Creating data.csv file for statistics...");
	{
		char aFilename[IO_MAX_PATH_LENGTH];
		sprintf_s(aFilename, sizeof(aFilename), "lr%.1embs%lldppoe%lldbots%drpb%d.csv", model_manager->GetLearningRate(), model_manager->GetMiniBatchSize(), model_manager->GetCountPPOEpochs(), count_bots, update_tick);
		logger.open("train\\" + dir_name + "\\" + aFilename);
		logger << "Step,Average reward,TPS,Dies,Average distance,Training loss,Actor loss,Critic loss,Learning rate,Time since start,Time to decide,Time to tick,Time rest,Time pre forward,Time forward,Time normal,Time to cpu,Time process last,Average speed" << endl;
	}
	dbg_msg("neuralnetwork", "data.csv file created and initialized.");

	// Recording demo to spectate how model performs
	string path_demo;
	{
		if(model_manager->IsTraining())
		{
			char aFilename[IO_MAX_PATH_LENGTH];
			str_format(aFilename, sizeof(aFilename), "%s_%d_%llu.demo", m_pServer->m_aCurrentMap, m_pServer->m_NetServer.Address().port, time_get());
			path_demo = "train/" + dir_name + "/demos/" + aFilename;
			int ret = m_pServer->m_aDemoRecorder[MAX_CLIENTS].Start(m_pStorage, m_pConsole, path_demo.c_str(), m_pGameContext->NetVersion(), m_pServer->m_aCurrentMap, &m_pServer->m_aCurrentMapSha256[CServer::MAP_TYPE_SIX], m_pServer->m_aCurrentMapCrc[CServer::MAP_TYPE_SIX], "server", m_pServer->m_aCurrentMapSize[CServer::MAP_TYPE_SIX], m_pServer->m_apCurrentMapData[CServer::MAP_TYPE_SIX]);
		}
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
	if(m_pServer->Tick() > 1000)
	{
		m_pServer->m_aDemoRecorder[MAX_CLIENTS].Stop();
	}
	// Handle bots
	if(m_pServer->Tick() % skip_tick == 0)
	{
		gamelayer = m_pGameContext->Layers()->GameLayer();
		pTiles = static_cast<CTile *>(Kernel()->RequestInterface<IMap>()->GetData(gamelayer->m_Data));
		int map_width = gamelayer->m_Width;
		int map_height = gamelayer->m_Height;

		//  apply new input
		// decide_time = time_get_impl();
		for(size_t i = 0; i < vBots.size(); i++)
		{
			auto bot = vBots[i];

			auto bot_character = bot->GetCharacter();
			CCharacterCore *bot_character_core;
			// auto bot_2_character = gamecontext->GetPlayerChar(bot_2->GetCID());
			if(respawn_all && model_manager->IsTraining())
			{
				vec2 spawn_pos;

				if(i % 3 == 2)
					spawn_pos = ball_spawn_pos;
				else
					spawn_pos = i % 3 == 0 ? first_bot_spawn_pos : second_bot_spawn_pos;
				
				bot->KillCharacter();
				bot->TryRespawn(spawn_pos);

				bot_character = bot->GetCharacter();
				bot_character_core = bot_character->Core();

				vBotsCumulativeRewards[i - ((i+1)/3)] = 0;
			}

			if(!bot_character || i % 3 == 2)
			{
				continue;
			}

			bot_character_core = bot_character->Core();

			int enemy_id = i % 3 == 0 ? i + 1 : i - 1 ;
			int ball_id = i % 3 == 0 ? i + 2 : i + 1;

			int rotate = i % 3 == 1 ? -1 : 1;

			CCharacterCore *enemy_character_core = vBots[enemy_id]->GetCharacter()->Core();

			CCharacterCore *ball_character_core = vBots[ball_id]->GetCharacter()->Core();

			vec2 bot_pos = bot_character_core->m_Pos;

			vec2 enemy_pos = enemy_character_core->m_Pos;

			vec2 ball_pos = ball_character_core->m_Pos;

			bool out_of_area = false;

			ModelInputInputs *input_inputs = &vInputInputs[i - ((i + 1) / 3)];

			// Local bot

			input_inputs->bot_pos = 
					{(bot_pos.x - volleyball_area_center.x) / (volleyball_area_sizes.x / 2.f),
					(bot_pos.y - volleyball_area_center.y) / (volleyball_area_sizes.y / 2.f)};
			input_inputs->bot_pos.x *= rotate;
			if(input_inputs->bot_pos.x < -1.f || input_inputs->bot_pos.x > 1.f
				|| input_inputs->bot_pos.y < -1.f || input_inputs->bot_pos.y > 1.f)
			{
				input_inputs->bot_pos = {0, 0};
				input_inputs->bot_is_out_of_area = true;
			}
			else
				input_inputs->bot_is_out_of_area = false;
			input_inputs->bot_vel = bot_character_core->m_Vel / 40.f;
			input_inputs->bot_vel.x *= rotate;

			input_inputs->bot_is_hooking = bot_character_core->m_HookState == HOOK_FLYING || bot_character_core->m_HookState == HOOK_GRABBED;
			input_inputs->bot_is_grabbed = bot_character_core->m_HookState == HOOK_GRABBED;
			input_inputs->bot_is_retracted = bot_character_core->m_HookState == HOOK_RETRACTED || (bot_character_core->m_HookState >= HOOK_RETRACT_START && bot_character_core->m_HookState <= HOOK_RETRACT_END);

			if(!input_inputs->bot_is_out_of_area && input_inputs->bot_is_hooking)
			{
				auto hook_relative = (bot_character_core->m_HookPos - bot_character_core->m_Pos) / m_pGameContext->Tuning()->m_HookLength;

				input_inputs->bot_hook_pos = (bot_character_core->m_HookPos - volleyball_area_start) / volleyball_area_sizes;
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

			input_inputs->enemy_pos =
				{(enemy_pos.x - volleyball_area_center.x) / (volleyball_area_sizes.x / 2.f),
					(enemy_pos.y - volleyball_area_center.y) / (volleyball_area_sizes.y / 2.f)};
			input_inputs->enemy_pos.x *= rotate;
			if(input_inputs->enemy_pos.x < -1.f || input_inputs->enemy_pos.x > 1.f || input_inputs->enemy_pos.y < -1.f || input_inputs->enemy_pos.y > 1.f)
			{
				input_inputs->enemy_pos = {0, 0};
				input_inputs->enemy_is_out_of_area = true;
			}
			else
				input_inputs->enemy_is_out_of_area = false;
			input_inputs->enemy_vel = enemy_character_core->m_Vel / 40.f;
			input_inputs->enemy_vel.x *= rotate;

			input_inputs->enemy_is_hooking = enemy_character_core->m_HookState == HOOK_FLYING || enemy_character_core->m_HookState == HOOK_GRABBED;
			input_inputs->enemy_is_grabbed = enemy_character_core->m_HookState == HOOK_GRABBED;
			input_inputs->enemy_is_retracted = enemy_character_core->m_HookState == HOOK_RETRACTED || (enemy_character_core->m_HookState >= HOOK_RETRACT_START && enemy_character_core->m_HookState <= HOOK_RETRACT_END);

			if(!input_inputs->enemy_is_out_of_area && input_inputs->enemy_is_hooking)
			{
				auto hook_relative = (enemy_character_core->m_HookPos - enemy_character_core->m_Pos) / m_pGameContext->Tuning()->m_HookLength;

				input_inputs->enemy_hook_pos = (enemy_character_core->m_HookPos - volleyball_area_start) / volleyball_area_sizes;
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
			//std::cout << ball_pos.x << std::endl;
			input_inputs->ball_pos =
				{(ball_pos.x - volleyball_area_center.x) / (volleyball_area_sizes.x / 2.f),
					(ball_pos.y - volleyball_area_center.y) / (volleyball_area_sizes.y / 2.f)};
			
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
			vBotLastPos[i - ((i + 1) / 3)] = bot_pos;
			vBotLastVel[i - ((i + 1) / 3)] = length(bot_character_core->m_Vel);
		}
		if(respawn_all)
		{
			respawn_all = false;
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

		vOutputs = model_manager->Decide(vInputInputs, time_pre_forward, time_forward, time_normal, time_to_cpu, time_process_last);

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
		auto bot = vBots[i];

		bot->UpdatePlaytime();

		if(i%3 == 2)
		{
			continue;
		}

		//  Move bot wherever you want
		// m_Direction:
		// 1 - Right
		// 0 - Stay
		// -1 - Left

		int angle = 0;

		auto bot_character = bot->GetCharacter();
		int model_jump = 0;
		vec2 model_angle;
		int model_direction = 0;
		int model_hook = 0;

		if(bot_character != nullptr)
		{
			ModelOutput returned_model = vOutputs[i - ((i+1)/3)]; // {0.0f}; //
			model_angle = returned_model.angle * 299.f;
			model_direction = returned_model.direction;
			model_hook = returned_model.hook;
		}

		CNetObj_PlayerInput pApplyInput;
		mem_zero(&pApplyInput, sizeof(pApplyInput));
		pApplyInput.m_TargetX = (int)round(model_angle.x);
		pApplyInput.m_TargetY = (int)round(model_angle.y);
		//pApplyInput.m_Jump = model_jump;
		pApplyInput.m_Direction = model_direction;
		pApplyInput.m_Hook = model_hook;

		m_pGameContext->OnClientPredictedInput(bot->GetCID(), &pApplyInput);
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
	static float cumulative_speed = 0;
	static int count_updated = 0;

	// Rewards
	static float goal_reward = 5.f;
	static float ball_on_side_reward = 0.1f;
	static float step_reward = -0.01f;
	static float divide_reward_by = 5.f;

	static int long_no_improvements_ticks = 200;

	//static std::vector<float> rewards;
	static float best_average = -999999.f;
	static float last_saved = -999999.f;

	auto now = std::chrono::high_resolution_clock::now();
	cumulative_time_to_tick += time_to_tick * 1000.f;

	// size_t summerr = 0;
	// decide_time = time_get_impl();
	if(m_pServer->Tick() % skip_tick == 0)
	{
		for(size_t team_id = 0; team_id < count_teams; team_id++)
		{
			bool match_is_done = false;

			int first_bot_id = team_id * 3;
			int second_bot_id = team_id * 3 + 1;
			int ball_id = team_id * 3 + 2;

			int teleport_num = 0;

			auto first_bot_character = vBots[first_bot_id]->GetCharacter();
			auto second_bot_character = vBots[second_bot_id]->GetCharacter();
			auto ball_character = vBots[ball_id]->GetCharacter();

			teleport_num = ball_character->teleport_num;
			ball_character->teleport_num = 0;

			auto first_bot_pos = first_bot_character->m_Pos;
			auto second_bot_pos = second_bot_character->m_Pos;
			auto ball_pos = ball_character->m_Pos;

			float reward = 0;

			if(teleport_num > 0)
			{
				std::cout << "Teleport num: " << teleport_num << std::endl;
				reward += teleport_num == 1 ? -goal_reward : goal_reward;
			}

			if(first_bot_pos.x < volleyball_final_area_right_top_corner.x && first_bot_pos.y > volleyball_final_area_right_top_corner.y \
				|| second_bot_pos.x < volleyball_final_area_right_top_corner.x && second_bot_pos.y > volleyball_final_area_right_top_corner.y)
			{
				match_is_done = true;

				// Respawn everyone in team
				for(size_t i = 0; i < 3; i++)
				{
					int bot_id = team_id * 3 + i;
					auto bot = vBots[bot_id];

					bot->KillCharacter();

					vec2 spawn_pos;

					if(i % 3 == 2)
					{
						spawn_pos = ball_spawn_pos;
					}
					else
					{
						spawn_pos = i % 3 == 0 ? first_bot_spawn_pos : second_bot_spawn_pos;
					}
					bot->TryRespawn(spawn_pos);

					vBotsCumulativeRewards[bot_id - ((bot_id + 1) / 3)] = 0;
				}
			}
			else
			{
				if(ball_pos.x > volleyball_area_start.x && ball_pos.y > volleyball_area_start.y && ball_pos.x < volleyball_area_end.x && ball_pos.y < volleyball_area_end.y)
				{
					if(ball_pos.x < volleyball_area_center.x)
					{
						reward -= ball_on_side_reward;
					}
					else if (ball_pos.x > volleyball_area_center.x)
					{
						reward += ball_on_side_reward;
					}
				}
			}

			float first_bot_reward = reward - step_reward;
			float second_bot_reward = reward - step_reward;

			model_manager->Reward(first_bot_reward / divide_reward_by, match_is_done);
			model_manager->Reward(second_bot_reward / divide_reward_by, match_is_done);
		}
		bool is_full = false;
		// cout << "Time rewards: " << (float)(time_get_impl() - decide_time) / (float)time_freq() << endl;
		//printf("4444\n");

		model_manager->SaveReplays(is_full);

		ticks_collected += 1;

		if(is_full)
		{
			// decide_time = time_get_impl();

			float avg_reward = cumulative_reward / (float)(dies + count_player_bots);
			float avg_dist = ((float)moved_distance / (float)(dies));
			float avg_speed = ((float)moved_distance / (float)((m_pServer->Tick() - last_update_tick) * count_player_bots) * (float)SERVER_TICK_SPEED);
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
			last_update_tick = m_pServer->Tick();
			// cout << "Time update: " << (float)(time_get_impl() - decide_time) / (float)time_freq() << endl;
			if(updated)
			{
				count_updated += 1;
				//rewards.clear();
				if(avg_dist - last_saved > last_saved * 0.1f && avg_dist - last_saved > 25)
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
				}
				else
				{
					/*if(was_recording)
					{
						Storage()->RemoveFile(demo_recorder->GetCurrentFilename(), IStorage::TYPE_ABSOLUTE);
					}*/
				}

				model_manager->Save("train\\" + dir_name + "\\models\\last");

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
						<< "," << avg_speed
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