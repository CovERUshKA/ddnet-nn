/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */

#include "server.h"

//#include <filesystem>
#include <iostream>
#include <base/logger.h>
#include <base/math.h>
#include <base/system.h>

#include <engine/config.h>
#include <engine/console.h>
#include <engine/engine.h>
#include <engine/map.h>
#include <engine/server.h>
#include <engine/storage.h>

#include <engine/shared/compression.h>
#include <engine/shared/config.h>
#include <engine/shared/console.h>
#include <engine/shared/demo.h>
#include <engine/shared/econ.h>
#include <engine/shared/fifo.h>
#include <engine/shared/filecollection.h>
#include <engine/shared/http.h>
#include <engine/shared/json.h>
#include <engine/shared/masterserver.h>
#include <engine/shared/netban.h>
#include <engine/shared/network.h>
#include <engine/shared/packer.h>
#include <engine/shared/protocol.h>
#include <engine/shared/protocol_ex.h>
#include <engine/shared/rust_version.h>
#include <engine/shared/snapshot.h>

#include <game/version.h>
#include <game/server/entities/character.h>
#include <game/server/gamecontext.h>
#include <game/server/gamecontroller.h>
#include <game/mapitems.h>

// DDRace
#include <engine/shared/linereader.h>
#include <vector>
#include <zlib.h>

#include "databases/connection.h"
#include "databases/connection_pool.h"
#include "register.h"

// Neural network
#include <game/server/player.h>
#include <engine/server/NN/ModelManager.h>
#include <engine/server/NN/AStar.h>
#include <numeric>
#include <random>
#include <fstream>
//#include <EvolutionNet/EvolutionNet.hpp>

extern bool IsInterrupted();

CSnapIDPool::CSnapIDPool()
{
	Reset();
}

void CSnapIDPool::Reset()
{
	for(int i = 0; i < MAX_IDS; i++)
	{
		m_aIDs[i].m_Next = i + 1;
		m_aIDs[i].m_State = ID_FREE;
	}

	m_aIDs[MAX_IDS - 1].m_Next = -1;
	m_FirstFree = 0;
	m_FirstTimed = -1;
	m_LastTimed = -1;
	m_Usage = 0;
	m_InUsage = 0;
}

void CSnapIDPool::RemoveFirstTimeout()
{
	int NextTimed = m_aIDs[m_FirstTimed].m_Next;

	// add it to the free list
	m_aIDs[m_FirstTimed].m_Next = m_FirstFree;
	m_aIDs[m_FirstTimed].m_State = ID_FREE;
	m_FirstFree = m_FirstTimed;

	// remove it from the timed list
	m_FirstTimed = NextTimed;
	if(m_FirstTimed == -1)
		m_LastTimed = -1;

	m_Usage--;
}

int CSnapIDPool::NewID()
{
	int64_t Now = time_get();

	// process timed ids
	while(m_FirstTimed != -1 && m_aIDs[m_FirstTimed].m_Timeout < Now)
		RemoveFirstTimeout();

	int ID = m_FirstFree;
	if(ID == -1)
	{
		dbg_msg("server", "invalid id");
		return ID;
	}
	m_FirstFree = m_aIDs[m_FirstFree].m_Next;
	m_aIDs[ID].m_State = ID_ALLOCATED;
	m_Usage++;
	m_InUsage++;
	return ID;
}

void CSnapIDPool::TimeoutIDs()
{
	// process timed ids
	while(m_FirstTimed != -1)
		RemoveFirstTimeout();
}

void CSnapIDPool::FreeID(int ID)
{
	if(ID < 0)
		return;
	dbg_assert((size_t)ID < std::size(m_aIDs), "id is out of range");
	dbg_assert(m_aIDs[ID].m_State == ID_ALLOCATED, "id is not allocated");

	m_InUsage--;
	m_aIDs[ID].m_State = ID_TIMED;
	m_aIDs[ID].m_Timeout = time_get() + time_freq() * 5;
	m_aIDs[ID].m_Next = -1;

	if(m_LastTimed != -1)
	{
		m_aIDs[m_LastTimed].m_Next = ID;
		m_LastTimed = ID;
	}
	else
	{
		m_FirstTimed = ID;
		m_LastTimed = ID;
	}
}

void CServerBan::InitServerBan(IConsole *pConsole, IStorage *pStorage, CServer *pServer)
{
	CNetBan::Init(pConsole, pStorage);

	m_pServer = pServer;

	// overwrites base command, todo: improve this
	Console()->Register("ban", "s[ip|id] ?i[minutes] r[reason]", CFGFLAG_SERVER | CFGFLAG_STORE, ConBanExt, this, "Ban player with ip/client id for x minutes for any reason");
	Console()->Register("ban_region", "s[region] s[ip|id] ?i[minutes] r[reason]", CFGFLAG_SERVER | CFGFLAG_STORE, ConBanRegion, this, "Ban player in a region");
	Console()->Register("ban_region_range", "s[region] s[first ip] s[last ip] ?i[minutes] r[reason]", CFGFLAG_SERVER | CFGFLAG_STORE, ConBanRegionRange, this, "Ban range in a region");
}

template<class T>
int CServerBan::BanExt(T *pBanPool, const typename T::CDataType *pData, int Seconds, const char *pReason)
{
	// validate address
	if(Server()->m_RconClientID >= 0 && Server()->m_RconClientID < MAX_CLIENTS &&
		Server()->m_aClients[Server()->m_RconClientID].m_State != CServer::CClient::STATE_EMPTY)
	{
		if(NetMatch(pData, Server()->m_NetServer.ClientAddr(Server()->m_RconClientID)))
		{
			Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "net_ban", "ban error (you can't ban yourself)");
			return -1;
		}

		for(int i = 0; i < MAX_CLIENTS; ++i)
		{
			if(i == Server()->m_RconClientID || Server()->m_aClients[i].m_State == CServer::CClient::STATE_EMPTY)
				continue;

			if(Server()->m_aClients[i].m_Authed >= Server()->m_RconAuthLevel && NetMatch(pData, Server()->m_NetServer.ClientAddr(i)))
			{
				Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "net_ban", "ban error (command denied)");
				return -1;
			}
		}
	}
	else if(Server()->m_RconClientID == IServer::RCON_CID_VOTE)
	{
		for(int i = 0; i < MAX_CLIENTS; ++i)
		{
			if(Server()->m_aClients[i].m_State == CServer::CClient::STATE_EMPTY)
				continue;

			if(Server()->m_aClients[i].m_Authed != AUTHED_NO && NetMatch(pData, Server()->m_NetServer.ClientAddr(i)))
			{
				Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "net_ban", "ban error (command denied)");
				return -1;
			}
		}
	}

	int Result = Ban(pBanPool, pData, Seconds, pReason);
	if(Result != 0)
		return Result;

	// drop banned clients
	typename T::CDataType Data = *pData;
	for(int i = 0; i < MAX_CLIENTS; ++i)
	{
		if(Server()->m_aClients[i].m_State == CServer::CClient::STATE_EMPTY)
			continue;

		if(NetMatch(&Data, Server()->m_NetServer.ClientAddr(i)))
		{
			CNetHash NetHash(&Data);
			char aBuf[256];
			MakeBanInfo(pBanPool->Find(&Data, &NetHash), aBuf, sizeof(aBuf), MSGTYPE_PLAYER);
			Server()->m_NetServer.Drop(i, aBuf);
		}
	}

	return Result;
}

int CServerBan::BanAddr(const NETADDR *pAddr, int Seconds, const char *pReason)
{
	return BanExt(&m_BanAddrPool, pAddr, Seconds, pReason);
}

int CServerBan::BanRange(const CNetRange *pRange, int Seconds, const char *pReason)
{
	if(pRange->IsValid())
		return BanExt(&m_BanRangePool, pRange, Seconds, pReason);

	Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "net_ban", "ban failed (invalid range)");
	return -1;
}

void CServerBan::ConBanExt(IConsole::IResult *pResult, void *pUser)
{
	CServerBan *pThis = static_cast<CServerBan *>(pUser);

	const char *pStr = pResult->GetString(0);
	int Minutes = pResult->NumArguments() > 1 ? clamp(pResult->GetInteger(1), 0, 525600) : 10;
	const char *pReason = pResult->NumArguments() > 2 ? pResult->GetString(2) : "Follow the server rules. Type /rules into the chat.";

	if(str_isallnum(pStr))
	{
		int ClientID = str_toint(pStr);
		if(ClientID < 0 || ClientID >= MAX_CLIENTS || pThis->Server()->m_aClients[ClientID].m_State == CServer::CClient::STATE_EMPTY)
			pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "net_ban", "ban error (invalid client id)");
		else
			pThis->BanAddr(pThis->Server()->m_NetServer.ClientAddr(ClientID), Minutes * 60, pReason);
	}
	else
		ConBan(pResult, pUser);
}

void CServerBan::ConBanRegion(IConsole::IResult *pResult, void *pUser)
{
	const char *pRegion = pResult->GetString(0);
	if(str_comp_nocase(pRegion, g_Config.m_SvRegionName))
		return;

	pResult->RemoveArgument(0);
	ConBanExt(pResult, pUser);
}

void CServerBan::ConBanRegionRange(IConsole::IResult *pResult, void *pUser)
{
	CServerBan *pServerBan = static_cast<CServerBan *>(pUser);

	const char *pRegion = pResult->GetString(0);
	if(str_comp_nocase(pRegion, g_Config.m_SvRegionName))
		return;

	pResult->RemoveArgument(0);
	ConBanRange(pResult, static_cast<CNetBan *>(pServerBan));
}

// Not thread-safe!
class CRconClientLogger : public ILogger
{
	CServer *m_pServer;
	int m_ClientID;

public:
	CRconClientLogger(CServer *pServer, int ClientID) :
		m_pServer(pServer),
		m_ClientID(ClientID)
	{
	}
	void Log(const CLogMessage *pMessage) override;
};

void CRconClientLogger::Log(const CLogMessage *pMessage)
{
	m_pServer->SendRconLogLine(m_ClientID, pMessage);
}

void CServer::CClient::Reset()
{
	// reset input
	for(auto &Input : m_aInputs)
		Input.m_GameTick = -1;
	m_CurrentInput = 0;
	mem_zero(&m_LatestInput, sizeof(m_LatestInput));

	m_Snapshots.PurgeAll();
	m_LastAckedSnapshot = -1;
	m_LastInputTick = -1;
	m_SnapRate = CClient::SNAPRATE_INIT;
	m_Score = 0;
	m_NextMapChunk = 0;
	m_Flags = 0;
}

CServer::CServer()
{
	m_pConfig = &g_Config;
	for(int i = 0; i < MAX_CLIENTS; i++)
		m_aDemoRecorder[i] = CDemoRecorder(&m_SnapshotDelta, true);
	m_aDemoRecorder[MAX_CLIENTS] = CDemoRecorder(&m_SnapshotDelta, false);

	m_TickSpeed = SERVER_TICK_SPEED;

	m_pGameServer = 0;

	m_CurrentGameTick = 0;
	m_RunServer = UNINITIALIZED;

	m_aShutdownReason[0] = 0;

	for(int i = 0; i < NUM_MAP_TYPES; i++)
	{
		m_apCurrentMapData[i] = 0;
		m_aCurrentMapSize[i] = 0;
	}

	m_MapReload = false;
	m_ReloadedWhenEmpty = false;
	m_aCurrentMap[0] = '\0';

	m_RconClientID = IServer::RCON_CID_SERV;
	m_RconAuthLevel = AUTHED_ADMIN;

	m_ServerInfoFirstRequest = 0;
	m_ServerInfoNumRequests = 0;
	m_ServerInfoNeedsUpdate = false;

#ifdef CONF_FAMILY_UNIX
	m_ConnLoggingSocketCreated = false;
#endif

	m_pConnectionPool = new CDbConnectionPool();
	m_pRegister = nullptr;

	m_aErrorShutdownReason[0] = 0;

	Init();
}

CServer::~CServer()
{
	for(auto &pCurrentMapData : m_apCurrentMapData)
	{
		free(pCurrentMapData);
	}

	if(m_RunServer != UNINITIALIZED)
	{
		for(auto &Client : m_aClients)
		{
			free(Client.m_pPersistentData);
		}
	}

	delete m_pRegister;
	delete m_pConnectionPool;
}

bool CServer::IsClientNameAvailable(int ClientID, const char *pNameRequest)
{
	// check for empty names
	if(!pNameRequest[0])
		return false;

	// check for names starting with /, as they can be abused to make people
	// write chat commands
	if(pNameRequest[0] == '/')
		return false;

	// make sure that two clients don't have the same name
	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(i != ClientID && m_aClients[i].m_State >= CClient::STATE_READY)
		{
			if(str_utf8_comp_confusable(pNameRequest, m_aClients[i].m_aName) == 0)
				return false;
		}
	}

	return true;
}

bool CServer::SetClientNameImpl(int ClientID, const char *pNameRequest, bool Set)
{
	dbg_assert(0 <= ClientID && ClientID < MAX_CLIENTS, "invalid client id");
	if(m_aClients[ClientID].m_State < CClient::STATE_READY)
		return false;

	CNameBan *pBanned = IsNameBanned(pNameRequest, m_vNameBans);
	if(pBanned)
	{
		if(m_aClients[ClientID].m_State == CClient::STATE_READY && Set)
		{
			char aBuf[256];
			if(pBanned->m_aReason[0])
			{
				str_format(aBuf, sizeof(aBuf), "Kicked (your name is banned: %s)", pBanned->m_aReason);
			}
			else
			{
				str_copy(aBuf, "Kicked (your name is banned)");
			}
			Kick(ClientID, aBuf);
		}
		return false;
	}

	// trim the name
	char aTrimmedName[MAX_NAME_LENGTH];
	str_copy(aTrimmedName, str_utf8_skip_whitespaces(pNameRequest));
	str_utf8_trim_right(aTrimmedName);

	char aNameTry[MAX_NAME_LENGTH];
	str_copy(aNameTry, aTrimmedName);

	if(!IsClientNameAvailable(ClientID, aNameTry))
	{
		// auto rename
		for(int i = 1;; i++)
		{
			str_format(aNameTry, sizeof(aNameTry), "(%d)%s", i, aTrimmedName);
			if(IsClientNameAvailable(ClientID, aNameTry))
				break;
		}
	}

	bool Changed = str_comp(m_aClients[ClientID].m_aName, aNameTry) != 0;

	if(Set)
	{
		// set the client name
		str_copy(m_aClients[ClientID].m_aName, aNameTry);
	}

	return Changed;
}

bool CServer::WouldClientNameChange(int ClientID, const char *pNameRequest)
{
	return SetClientNameImpl(ClientID, pNameRequest, false);
}

void CServer::SetClientName(int ClientID, const char *pName)
{
	SetClientNameImpl(ClientID, pName, true);
}

void CServer::SetClientClan(int ClientID, const char *pClan)
{
	if(ClientID < 0 || ClientID >= MAX_CLIENTS || m_aClients[ClientID].m_State < CClient::STATE_READY || !pClan)
		return;

	str_copy(m_aClients[ClientID].m_aClan, pClan);
}

void CServer::SetClientCountry(int ClientID, int Country)
{
	if(ClientID < 0 || ClientID >= MAX_CLIENTS || m_aClients[ClientID].m_State < CClient::STATE_READY)
		return;

	m_aClients[ClientID].m_Country = Country;
}

void CServer::SetClientScore(int ClientID, int Score)
{
	if(ClientID < 0 || ClientID >= MAX_CLIENTS || m_aClients[ClientID].m_State < CClient::STATE_READY)
		return;

	if(m_aClients[ClientID].m_Score != Score)
		ExpireServerInfo();

	m_aClients[ClientID].m_Score = Score;
}

void CServer::SetClientFlags(int ClientID, int Flags)
{
	if(ClientID < 0 || ClientID >= MAX_CLIENTS || m_aClients[ClientID].m_State < CClient::STATE_READY)
		return;

	m_aClients[ClientID].m_Flags = Flags;
}

void CServer::Kick(int ClientID, const char *pReason)
{
	if(ClientID < 0 || ClientID >= MAX_CLIENTS || m_aClients[ClientID].m_State == CClient::STATE_EMPTY)
	{
		Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", "invalid client id to kick");
		return;
	}
	else if(m_RconClientID == ClientID)
	{
		Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", "you can't kick yourself");
		return;
	}
	else if(m_aClients[ClientID].m_Authed > m_RconAuthLevel)
	{
		Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", "kick command denied");
		return;
	}

	m_NetServer.Drop(ClientID, pReason);
}

void CServer::Ban(int ClientID, int Seconds, const char *pReason)
{
	NETADDR Addr;
	GetClientAddr(ClientID, &Addr);
	m_NetServer.NetBan()->BanAddr(&Addr, Seconds, pReason);
}

int64_t CServer::TickStartTime(int Tick)
{
	return m_GameStartTime + (time_freq() * Tick) / SERVER_TICK_SPEED;
}

int CServer::Init()
{
	for(auto &Client : m_aClients)
	{
		Client.m_State = CClient::STATE_EMPTY;
		Client.m_aName[0] = 0;
		Client.m_aClan[0] = 0;
		Client.m_Country = -1;
		Client.m_Snapshots.Init();
		Client.m_Traffic = 0;
		Client.m_TrafficSince = 0;
		Client.m_ShowIps = false;
		Client.m_AuthKey = -1;
		Client.m_Latency = 0;
		Client.m_Sixup = false;
	}

	m_CurrentGameTick = 0;

	m_AnnouncementLastLine = 0;
	mem_zero(m_aPrevStates, sizeof(m_aPrevStates));

	return 0;
}

void CServer::SendLogLine(const CLogMessage *pMessage)
{
	if(pMessage->m_Level <= IConsole::ToLogLevel(g_Config.m_ConsoleOutputLevel))
	{
		SendRconLogLine(-1, pMessage);
	}
	if(pMessage->m_Level <= IConsole::ToLogLevel(g_Config.m_EcOutputLevel))
	{
		m_Econ.Send(-1, pMessage->m_aLine);
	}
}

void CServer::SetRconCID(int ClientID)
{
	m_RconClientID = ClientID;
}

int CServer::GetAuthedState(int ClientID) const
{
	return m_aClients[ClientID].m_Authed;
}

const char *CServer::GetAuthName(int ClientID) const
{
	int Key = m_aClients[ClientID].m_AuthKey;
	if(Key == -1)
	{
		return 0;
	}
	return m_AuthManager.KeyIdent(Key);
}

bool CServer::GetClientInfo(int ClientID, CClientInfo *pInfo) const
{
	dbg_assert(ClientID >= 0 && ClientID < MAX_CLIENTS, "ClientID is not valid");
	dbg_assert(pInfo != nullptr, "pInfo cannot be null");

	if(m_aClients[ClientID].m_State == CClient::STATE_INGAME)
	{
		pInfo->m_pName = m_aClients[ClientID].m_aName;
		pInfo->m_Latency = m_aClients[ClientID].m_Latency;
		pInfo->m_GotDDNetVersion = m_aClients[ClientID].m_DDNetVersionSettled;
		pInfo->m_DDNetVersion = m_aClients[ClientID].m_DDNetVersion >= 0 ? m_aClients[ClientID].m_DDNetVersion : VERSION_VANILLA;
		if(m_aClients[ClientID].m_GotDDNetVersionPacket)
		{
			pInfo->m_pConnectionID = &m_aClients[ClientID].m_ConnectionID;
			pInfo->m_pDDNetVersionStr = m_aClients[ClientID].m_aDDNetVersionStr;
		}
		else
		{
			pInfo->m_pConnectionID = nullptr;
			pInfo->m_pDDNetVersionStr = nullptr;
		}
		return true;
	}
	return false;
}

void CServer::SetClientDDNetVersion(int ClientID, int DDNetVersion)
{
	dbg_assert(ClientID >= 0 && ClientID < MAX_CLIENTS, "ClientID is not valid");

	if(m_aClients[ClientID].m_State == CClient::STATE_INGAME)
	{
		m_aClients[ClientID].m_DDNetVersion = DDNetVersion;
		m_aClients[ClientID].m_DDNetVersionSettled = true;
	}
}

void CServer::GetClientAddr(int ClientID, char *pAddrStr, int Size) const
{
	if(ClientID >= 0 && ClientID < MAX_CLIENTS && m_aClients[ClientID].m_State == CClient::STATE_INGAME)
		net_addr_str(m_NetServer.ClientAddr(ClientID), pAddrStr, Size, false);
}

const char *CServer::ClientName(int ClientID) const
{
	if(ClientID < 0 || ClientID >= MAX_CLIENTS || m_aClients[ClientID].m_State == CServer::CClient::STATE_EMPTY)
		return "(invalid)";
	if(m_aClients[ClientID].m_State == CServer::CClient::STATE_INGAME)
		return m_aClients[ClientID].m_aName;
	else
		return "(connecting)";
}

const char *CServer::ClientClan(int ClientID) const
{
	if(ClientID < 0 || ClientID >= MAX_CLIENTS || m_aClients[ClientID].m_State == CServer::CClient::STATE_EMPTY)
		return "";
	if(m_aClients[ClientID].m_State == CServer::CClient::STATE_INGAME)
		return m_aClients[ClientID].m_aClan;
	else
		return "";
}

int CServer::ClientCountry(int ClientID) const
{
	if(ClientID < 0 || ClientID >= MAX_CLIENTS || m_aClients[ClientID].m_State == CServer::CClient::STATE_EMPTY)
		return -1;
	if(m_aClients[ClientID].m_State == CServer::CClient::STATE_INGAME)
		return m_aClients[ClientID].m_Country;
	else
		return -1;
}

bool CServer::ClientIngame(int ClientID) const
{
	return ClientID >= 0 && ClientID < MAX_CLIENTS && m_aClients[ClientID].m_State == CServer::CClient::STATE_INGAME;
}

bool CServer::ClientAuthed(int ClientID) const
{
	return ClientID >= 0 && ClientID < MAX_CLIENTS && m_aClients[ClientID].m_Authed;
}

int CServer::Port() const
{
	return m_NetServer.Address().port;
}

int CServer::MaxClients() const
{
	return m_RunServer == UNINITIALIZED ? 0 : m_NetServer.MaxClients();
}

int CServer::ClientCount() const
{
	int ClientCount = 0;
	for(const auto &Client : m_aClients)
	{
		if(Client.m_State != CClient::STATE_EMPTY)
		{
			ClientCount++;
		}
	}

	return ClientCount;
}

int CServer::DistinctClientCount() const
{
	NETADDR aAddresses[MAX_CLIENTS];
	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(m_aClients[i].m_State != CClient::STATE_EMPTY)
		{
			GetClientAddr(i, &aAddresses[i]);
		}
	}

	int ClientCount = 0;
	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(m_aClients[i].m_State != CClient::STATE_EMPTY)
		{
			ClientCount++;
			for(int j = 0; j < i; j++)
			{
				if(!net_addr_comp_noport(&aAddresses[i], &aAddresses[j]))
				{
					ClientCount--;
					break;
				}
			}
		}
	}

	return ClientCount;
}

int CServer::GetClientVersion(int ClientID) const
{
	// Assume latest client version for server demos
	if(ClientID == SERVER_DEMO_CLIENT)
		return CLIENT_VERSIONNR;

	CClientInfo Info;
	if(GetClientInfo(ClientID, &Info))
		return Info.m_DDNetVersion;
	return VERSION_NONE;
}

static inline bool RepackMsg(const CMsgPacker *pMsg, CPacker &Packer, bool Sixup)
{
	int MsgId = pMsg->m_MsgID;
	Packer.Reset();

	if(Sixup && !pMsg->m_NoTranslate)
	{
		if(pMsg->m_System)
		{
			if(MsgId >= OFFSET_UUID)
				;
			else if(MsgId >= NETMSG_MAP_CHANGE && MsgId <= NETMSG_MAP_DATA)
				;
			else if(MsgId >= NETMSG_CON_READY && MsgId <= NETMSG_INPUTTIMING)
				MsgId += 1;
			else if(MsgId == NETMSG_RCON_LINE)
				MsgId = 13;
			else if(MsgId >= NETMSG_AUTH_CHALLANGE && MsgId <= NETMSG_AUTH_RESULT)
				MsgId += 4;
			else if(MsgId >= NETMSG_PING && MsgId <= NETMSG_ERROR)
				MsgId += 4;
			else if(MsgId >= NETMSG_RCON_CMD_ADD && MsgId <= NETMSG_RCON_CMD_REM)
				MsgId -= 11;
			else
			{
				dbg_msg("net", "DROP send sys %d", MsgId);
				return true;
			}
		}
		else
		{
			if(MsgId >= 0 && MsgId < OFFSET_UUID)
				MsgId = Msg_SixToSeven(MsgId);

			if(MsgId < 0)
				return true;
		}
	}

	if(MsgId < OFFSET_UUID)
	{
		Packer.AddInt((MsgId << 1) | (pMsg->m_System ? 1 : 0));
	}
	else
	{
		Packer.AddInt(pMsg->m_System ? 1 : 0); // NETMSG_EX, NETMSGTYPE_EX
		g_UuidManager.PackUuid(MsgId, &Packer);
	}
	Packer.AddRaw(pMsg->Data(), pMsg->Size());

	return false;
}

int CServer::SendMsg(CMsgPacker *pMsg, int Flags, int ClientID)
{
	CNetChunk Packet;
	mem_zero(&Packet, sizeof(CNetChunk));
	if(Flags & MSGFLAG_VITAL)
		Packet.m_Flags |= NETSENDFLAG_VITAL;
	if(Flags & MSGFLAG_FLUSH)
		Packet.m_Flags |= NETSENDFLAG_FLUSH;

	if(ClientID < 0)
	{
		CPacker Pack6, Pack7;
		if(RepackMsg(pMsg, Pack6, false))
			return -1;
		if(RepackMsg(pMsg, Pack7, true))
			return -1;

		// write message to demo recorders
		if(!(Flags & MSGFLAG_NORECORD))
		{
			for(auto &Recorder : m_aDemoRecorder)
				if(Recorder.IsRecording())
					Recorder.RecordMessage(Pack6.Data(), Pack6.Size());
		}

		if(!(Flags & MSGFLAG_NOSEND))
		{
			for(int i = 0; i < MAX_CLIENTS; i++)
			{
				if(m_aClients[i].m_State == CClient::STATE_INGAME)
				{
					CPacker *pPack = m_aClients[i].m_Sixup ? &Pack7 : &Pack6;
					Packet.m_pData = pPack->Data();
					Packet.m_DataSize = pPack->Size();
					Packet.m_ClientID = i;
					if(Antibot()->OnEngineServerMessage(i, Packet.m_pData, Packet.m_DataSize, Flags))
					{
						continue;
					}
					m_NetServer.Send(&Packet);
				}
			}
		}
	}
	else
	{
		CPacker Pack;
		if(RepackMsg(pMsg, Pack, m_aClients[ClientID].m_Sixup))
			return -1;

		Packet.m_ClientID = ClientID;
		Packet.m_pData = Pack.Data();
		Packet.m_DataSize = Pack.Size();

		if(Antibot()->OnEngineServerMessage(ClientID, Packet.m_pData, Packet.m_DataSize, Flags))
		{
			return 0;
		}

		// write message to demo recorders
		if(!(Flags & MSGFLAG_NORECORD))
		{
			if(m_aDemoRecorder[ClientID].IsRecording())
				m_aDemoRecorder[ClientID].RecordMessage(Pack.Data(), Pack.Size());
			if(m_aDemoRecorder[MAX_CLIENTS].IsRecording())
				m_aDemoRecorder[MAX_CLIENTS].RecordMessage(Pack.Data(), Pack.Size());
		}

		if(!(Flags & MSGFLAG_NOSEND))
			m_NetServer.Send(&Packet);
	}

	return 0;
}

void CServer::SendMsgRaw(int ClientID, const void *pData, int Size, int Flags)
{
	CNetChunk Packet;
	mem_zero(&Packet, sizeof(CNetChunk));
	Packet.m_ClientID = ClientID;
	Packet.m_pData = pData;
	Packet.m_DataSize = Size;
	Packet.m_Flags = 0;
	if(Flags & MSGFLAG_VITAL)
	{
		Packet.m_Flags |= NETSENDFLAG_VITAL;
	}
	if(Flags & MSGFLAG_FLUSH)
	{
		Packet.m_Flags |= NETSENDFLAG_FLUSH;
	}
	m_NetServer.Send(&Packet);
}

void CServer::DoSnapshot()
{
	GameServer()->OnPreSnap();

	// create snapshot for demo recording
	if(m_aDemoRecorder[MAX_CLIENTS].IsRecording())
	{
		char aData[CSnapshot::MAX_SIZE];

		// build snap and possibly add some messages
		m_SnapshotBuilder.Init();
		GameServer()->OnSnap(-1);
		int SnapshotSize = m_SnapshotBuilder.Finish(aData);

		// write snapshot
		m_aDemoRecorder[MAX_CLIENTS].RecordSnapshot(Tick(), aData, SnapshotSize);
	}

	// create snapshots for all clients
	for(int i = 0; i < MaxClients(); i++)
	{
		// client must be ingame to receive snapshots
		if(m_aClients[i].m_State != CClient::STATE_INGAME)
			continue;

		// this client is trying to recover, don't spam snapshots
		//if(m_aClients[i].m_SnapRate == CClient::SNAPRATE_RECOVER && (Tick() % 50) != 0)
			//continue;

		// this client is trying to recover, don't spam snapshots
		//if(m_aClients[i].m_SnapRate == CClient::SNAPRATE_INIT && (Tick() % 10) != 0)
			//continue;

		{
			m_SnapshotBuilder.Init(m_aClients[i].m_Sixup);

			GameServer()->OnSnap(i);

			// finish snapshot
			char aData[CSnapshot::MAX_SIZE];
			CSnapshot *pData = (CSnapshot *)aData; // Fix compiler warning for strict-aliasing
			int SnapshotSize = m_SnapshotBuilder.Finish(pData);

			if(m_aDemoRecorder[i].IsRecording())
			{
				// write snapshot
				m_aDemoRecorder[i].RecordSnapshot(Tick(), aData, SnapshotSize);
			}

			int Crc = pData->Crc();

			// remove old snapshots
			// keep 3 seconds worth of snapshots
			m_aClients[i].m_Snapshots.PurgeUntil(m_CurrentGameTick - SERVER_TICK_SPEED * 3);

			// save the snapshot
			m_aClients[i].m_Snapshots.Add(m_CurrentGameTick, time_get(), SnapshotSize, pData, 0, nullptr);

			// find snapshot that we can perform delta against
			static CSnapshot s_EmptySnap;
			s_EmptySnap.Clear();

			int DeltaTick = -1;
			CSnapshot *pDeltashot = &s_EmptySnap;
			{
				int DeltashotSize = m_aClients[i].m_Snapshots.Get(m_aClients[i].m_LastAckedSnapshot, 0, &pDeltashot, 0);
				if(DeltashotSize >= 0)
					DeltaTick = m_aClients[i].m_LastAckedSnapshot;
				else
				{
					// no acked package found, force client to recover rate
					if(m_aClients[i].m_SnapRate == CClient::SNAPRATE_FULL)
						m_aClients[i].m_SnapRate = CClient::SNAPRATE_RECOVER;
				}
			}

			// create delta
			m_SnapshotDelta.SetStaticsize(protocol7::NETEVENTTYPE_SOUNDWORLD, m_aClients[i].m_Sixup);
			m_SnapshotDelta.SetStaticsize(protocol7::NETEVENTTYPE_DAMAGE, m_aClients[i].m_Sixup);
			char aDeltaData[CSnapshot::MAX_SIZE];
			int DeltaSize = m_SnapshotDelta.CreateDelta(pDeltashot, pData, aDeltaData);

			if(DeltaSize)
			{
				// compress it
				const int MaxSize = MAX_SNAPSHOT_PACKSIZE;

				char aCompData[CSnapshot::MAX_SIZE];
				SnapshotSize = CVariableInt::Compress(aDeltaData, DeltaSize, aCompData, sizeof(aCompData));
				int NumPackets = (SnapshotSize + MaxSize - 1) / MaxSize;

				for(int n = 0, Left = SnapshotSize; Left > 0; n++)
				{
					int Chunk = Left < MaxSize ? Left : MaxSize;
					Left -= Chunk;

					if(NumPackets == 1)
					{
						CMsgPacker Msg(NETMSG_SNAPSINGLE, true);
						Msg.AddInt(m_CurrentGameTick);
						Msg.AddInt(m_CurrentGameTick - DeltaTick);
						Msg.AddInt(Crc);
						Msg.AddInt(Chunk);
						Msg.AddRaw(&aCompData[n * MaxSize], Chunk);
						SendMsg(&Msg, MSGFLAG_FLUSH, i);
					}
					else
					{
						CMsgPacker Msg(NETMSG_SNAP, true);
						Msg.AddInt(m_CurrentGameTick);
						Msg.AddInt(m_CurrentGameTick - DeltaTick);
						Msg.AddInt(NumPackets);
						Msg.AddInt(n);
						Msg.AddInt(Crc);
						Msg.AddInt(Chunk);
						Msg.AddRaw(&aCompData[n * MaxSize], Chunk);
						SendMsg(&Msg, MSGFLAG_FLUSH, i);
					}
				}
			}
			else
			{
				CMsgPacker Msg(NETMSG_SNAPEMPTY, true);
				Msg.AddInt(m_CurrentGameTick);
				Msg.AddInt(m_CurrentGameTick - DeltaTick);
				SendMsg(&Msg, MSGFLAG_FLUSH, i);
			}
		}
	}

	GameServer()->OnPostSnap();
}

int CServer::ClientRejoinCallback(int ClientID, void *pUser)
{
	CServer *pThis = (CServer *)pUser;

	pThis->m_aClients[ClientID].m_Authed = AUTHED_NO;
	pThis->m_aClients[ClientID].m_AuthKey = -1;
	pThis->m_aClients[ClientID].m_pRconCmdToSend = 0;
	pThis->m_aClients[ClientID].m_DDNetVersion = VERSION_NONE;
	pThis->m_aClients[ClientID].m_GotDDNetVersionPacket = false;
	pThis->m_aClients[ClientID].m_DDNetVersionSettled = false;

	pThis->m_aClients[ClientID].Reset();

	pThis->SendMap(ClientID);

	return 0;
}

int CServer::NewClientNoAuthCallback(int ClientID, void *pUser)
{
	CServer *pThis = (CServer *)pUser;

	pThis->m_aClients[ClientID].m_DnsblState = CClient::DNSBL_STATE_NONE;

	pThis->m_aClients[ClientID].m_State = CClient::STATE_CONNECTING;
	pThis->m_aClients[ClientID].m_aName[0] = 0;
	pThis->m_aClients[ClientID].m_aClan[0] = 0;
	pThis->m_aClients[ClientID].m_Country = -1;
	pThis->m_aClients[ClientID].m_Authed = AUTHED_NO;
	pThis->m_aClients[ClientID].m_AuthKey = -1;
	pThis->m_aClients[ClientID].m_AuthTries = 0;
	pThis->m_aClients[ClientID].m_pRconCmdToSend = 0;
	pThis->m_aClients[ClientID].m_ShowIps = false;
	pThis->m_aClients[ClientID].m_DDNetVersion = VERSION_NONE;
	pThis->m_aClients[ClientID].m_GotDDNetVersionPacket = false;
	pThis->m_aClients[ClientID].m_DDNetVersionSettled = false;
	pThis->m_aClients[ClientID].Reset();

	pThis->SendCapabilities(ClientID);
	pThis->SendMap(ClientID);
#if defined(CONF_FAMILY_UNIX)
	pThis->SendConnLoggingCommand(OPEN_SESSION, pThis->m_NetServer.ClientAddr(ClientID));
#endif
	return 0;
}

int CServer::NewClientCallback(int ClientID, void *pUser, bool Sixup)
{
	CServer *pThis = (CServer *)pUser;
	pThis->m_aClients[ClientID].m_State = CClient::STATE_PREAUTH;
	pThis->m_aClients[ClientID].m_DnsblState = CClient::DNSBL_STATE_NONE;
	pThis->m_aClients[ClientID].m_aName[0] = 0;
	pThis->m_aClients[ClientID].m_aClan[0] = 0;
	pThis->m_aClients[ClientID].m_Country = -1;
	pThis->m_aClients[ClientID].m_Authed = AUTHED_NO;
	pThis->m_aClients[ClientID].m_AuthKey = -1;
	pThis->m_aClients[ClientID].m_AuthTries = 0;
	pThis->m_aClients[ClientID].m_pRconCmdToSend = 0;
	pThis->m_aClients[ClientID].m_Traffic = 0;
	pThis->m_aClients[ClientID].m_TrafficSince = 0;
	pThis->m_aClients[ClientID].m_ShowIps = false;
	pThis->m_aClients[ClientID].m_DDNetVersion = VERSION_NONE;
	pThis->m_aClients[ClientID].m_GotDDNetVersionPacket = false;
	pThis->m_aClients[ClientID].m_DDNetVersionSettled = false;
	mem_zero(&pThis->m_aClients[ClientID].m_Addr, sizeof(NETADDR));
	pThis->m_aClients[ClientID].Reset();

	pThis->GameServer()->OnClientEngineJoin(ClientID, Sixup);
	pThis->Antibot()->OnEngineClientJoin(ClientID, Sixup);

	pThis->m_aClients[ClientID].m_Sixup = Sixup;

#if defined(CONF_FAMILY_UNIX)
	pThis->SendConnLoggingCommand(OPEN_SESSION, pThis->m_NetServer.ClientAddr(ClientID));
#endif
	return 0;
}

void CServer::InitDnsbl(int ClientID)
{
	NETADDR Addr = *m_NetServer.ClientAddr(ClientID);

	//TODO: support ipv6
	if(Addr.type != NETTYPE_IPV4)
		return;

	// build dnsbl host lookup
	char aBuf[256];
	if(Config()->m_SvDnsblKey[0] == '\0')
	{
		// without key
		str_format(aBuf, sizeof(aBuf), "%d.%d.%d.%d.%s", Addr.ip[3], Addr.ip[2], Addr.ip[1], Addr.ip[0], Config()->m_SvDnsblHost);
	}
	else
	{
		// with key
		str_format(aBuf, sizeof(aBuf), "%s.%d.%d.%d.%d.%s", Config()->m_SvDnsblKey, Addr.ip[3], Addr.ip[2], Addr.ip[1], Addr.ip[0], Config()->m_SvDnsblHost);
	}

	IEngine *pEngine = Kernel()->RequestInterface<IEngine>();
	pEngine->AddJob(m_aClients[ClientID].m_pDnsblLookup = std::make_shared<CHostLookup>(aBuf, NETTYPE_IPV4));
	m_aClients[ClientID].m_DnsblState = CClient::DNSBL_STATE_PENDING;
}

#ifdef CONF_FAMILY_UNIX
void CServer::SendConnLoggingCommand(CONN_LOGGING_CMD Cmd, const NETADDR *pAddr)
{
	if(!Config()->m_SvConnLoggingServer[0] || !m_ConnLoggingSocketCreated)
		return;

	// pack the data and send it
	unsigned char aData[23] = {0};
	aData[0] = Cmd;
	mem_copy(&aData[1], &pAddr->type, 4);
	mem_copy(&aData[5], pAddr->ip, 16);
	mem_copy(&aData[21], &pAddr->port, 2);

	net_unix_send(m_ConnLoggingSocket, &m_ConnLoggingDestAddr, aData, sizeof(aData));
}
#endif

int CServer::DelClientCallback(int ClientID, const char *pReason, void *pUser)
{
	CServer *pThis = (CServer *)pUser;

	char aAddrStr[NETADDR_MAXSTRSIZE];
	net_addr_str(pThis->m_NetServer.ClientAddr(ClientID), aAddrStr, sizeof(aAddrStr), true);

	char aBuf[256];
	str_format(aBuf, sizeof(aBuf), "client dropped. cid=%d addr=<{%s}> reason='%s'", ClientID, aAddrStr, pReason);
	pThis->Console()->Print(IConsole::OUTPUT_LEVEL_ADDINFO, "server", aBuf);

	// notify the mod about the drop
	if(pThis->m_aClients[ClientID].m_State >= CClient::STATE_READY)
		pThis->GameServer()->OnClientDrop(ClientID, pReason);

	pThis->m_aClients[ClientID].m_State = CClient::STATE_EMPTY;
	pThis->m_aClients[ClientID].m_aName[0] = 0;
	pThis->m_aClients[ClientID].m_aClan[0] = 0;
	pThis->m_aClients[ClientID].m_Country = -1;
	pThis->m_aClients[ClientID].m_Authed = AUTHED_NO;
	pThis->m_aClients[ClientID].m_AuthKey = -1;
	pThis->m_aClients[ClientID].m_AuthTries = 0;
	pThis->m_aClients[ClientID].m_pRconCmdToSend = 0;
	pThis->m_aClients[ClientID].m_Traffic = 0;
	pThis->m_aClients[ClientID].m_TrafficSince = 0;
	pThis->m_aClients[ClientID].m_ShowIps = false;
	pThis->m_aPrevStates[ClientID] = CClient::STATE_EMPTY;
	pThis->m_aClients[ClientID].m_Snapshots.PurgeAll();
	pThis->m_aClients[ClientID].m_Sixup = false;

	pThis->GameServer()->OnClientEngineDrop(ClientID, pReason);
	pThis->Antibot()->OnEngineClientDrop(ClientID, pReason);
#if defined(CONF_FAMILY_UNIX)
	pThis->SendConnLoggingCommand(CLOSE_SESSION, pThis->m_NetServer.ClientAddr(ClientID));
#endif
	return 0;
}

void CServer::SendRconType(int ClientID, bool UsernameReq)
{
	CMsgPacker Msg(NETMSG_RCONTYPE, true);
	Msg.AddInt(UsernameReq);
	SendMsg(&Msg, MSGFLAG_VITAL, ClientID);
}

void CServer::GetMapInfo(char *pMapName, int MapNameSize, int *pMapSize, SHA256_DIGEST *pMapSha256, int *pMapCrc)
{
	str_copy(pMapName, GetMapName(), MapNameSize);
	*pMapSize = m_aCurrentMapSize[MAP_TYPE_SIX];
	*pMapSha256 = m_aCurrentMapSha256[MAP_TYPE_SIX];
	*pMapCrc = m_aCurrentMapCrc[MAP_TYPE_SIX];
}

void CServer::SendCapabilities(int ClientID)
{
	CMsgPacker Msg(NETMSG_CAPABILITIES, true);
	Msg.AddInt(SERVERCAP_CURVERSION); // version
	Msg.AddInt(SERVERCAPFLAG_DDNET | SERVERCAPFLAG_CHATTIMEOUTCODE | SERVERCAPFLAG_ANYPLAYERFLAG | SERVERCAPFLAG_PINGEX | SERVERCAPFLAG_ALLOWDUMMY | SERVERCAPFLAG_SYNCWEAPONINPUT); // flags
	SendMsg(&Msg, MSGFLAG_VITAL, ClientID);
}

void CServer::SendMap(int ClientID)
{
	int MapType = IsSixup(ClientID) ? MAP_TYPE_SIXUP : MAP_TYPE_SIX;
	{
		CMsgPacker Msg(NETMSG_MAP_DETAILS, true);
		Msg.AddString(GetMapName(), 0);
		Msg.AddRaw(&m_aCurrentMapSha256[MapType].data, sizeof(m_aCurrentMapSha256[MapType].data));
		Msg.AddInt(m_aCurrentMapCrc[MapType]);
		Msg.AddInt(m_aCurrentMapSize[MapType]);
		Msg.AddString("", 0); // HTTPS map download URL
		SendMsg(&Msg, MSGFLAG_VITAL, ClientID);
	}
	{
		CMsgPacker Msg(NETMSG_MAP_CHANGE, true);
		Msg.AddString(GetMapName(), 0);
		Msg.AddInt(m_aCurrentMapCrc[MapType]);
		Msg.AddInt(m_aCurrentMapSize[MapType]);
		if(MapType == MAP_TYPE_SIXUP)
		{
			Msg.AddInt(Config()->m_SvMapWindow);
			Msg.AddInt(1024 - 128);
			Msg.AddRaw(m_aCurrentMapSha256[MapType].data, sizeof(m_aCurrentMapSha256[MapType].data));
		}
		SendMsg(&Msg, MSGFLAG_VITAL | MSGFLAG_FLUSH, ClientID);
	}

	m_aClients[ClientID].m_NextMapChunk = 0;
}

void CServer::SendMapData(int ClientID, int Chunk)
{
	int MapType = IsSixup(ClientID) ? MAP_TYPE_SIXUP : MAP_TYPE_SIX;
	unsigned int ChunkSize = 1024 - 128;
	unsigned int Offset = Chunk * ChunkSize;
	int Last = 0;

	// drop faulty map data requests
	if(Chunk < 0 || Offset > m_aCurrentMapSize[MapType])
		return;

	if(Offset + ChunkSize >= m_aCurrentMapSize[MapType])
	{
		ChunkSize = m_aCurrentMapSize[MapType] - Offset;
		Last = 1;
	}

	CMsgPacker Msg(NETMSG_MAP_DATA, true);
	if(MapType == MAP_TYPE_SIX)
	{
		Msg.AddInt(Last);
		Msg.AddInt(m_aCurrentMapCrc[MAP_TYPE_SIX]);
		Msg.AddInt(Chunk);
		Msg.AddInt(ChunkSize);
	}
	Msg.AddRaw(&m_apCurrentMapData[MapType][Offset], ChunkSize);
	SendMsg(&Msg, MSGFLAG_VITAL | MSGFLAG_FLUSH, ClientID);

	if(Config()->m_Debug)
	{
		char aBuf[256];
		str_format(aBuf, sizeof(aBuf), "sending chunk %d with size %d", Chunk, ChunkSize);
		Console()->Print(IConsole::OUTPUT_LEVEL_DEBUG, "server", aBuf);
	}
}

void CServer::SendConnectionReady(int ClientID)
{
	CMsgPacker Msg(NETMSG_CON_READY, true);
	SendMsg(&Msg, MSGFLAG_VITAL | MSGFLAG_FLUSH, ClientID);
}

void CServer::SendRconLine(int ClientID, const char *pLine)
{
	CMsgPacker Msg(NETMSG_RCON_LINE, true);
	Msg.AddString(pLine, 512);
	SendMsg(&Msg, MSGFLAG_VITAL, ClientID);
}

void CServer::SendRconLogLine(int ClientID, const CLogMessage *pMessage)
{
	const char *pLine = pMessage->m_aLine;
	const char *pStart = str_find(pLine, "<{");
	const char *pEnd = pStart == NULL ? NULL : str_find(pStart + 2, "}>");
	const char *pLineWithoutIps;
	char aLine[512];
	char aLineWithoutIps[512];
	aLine[0] = '\0';
	aLineWithoutIps[0] = '\0';

	if(pStart == NULL || pEnd == NULL)
	{
		pLineWithoutIps = pLine;
	}
	else
	{
		str_append(aLine, pLine, pStart - pLine + 1);
		str_append(aLine, pStart + 2, pStart - pLine + pEnd - pStart - 1);
		str_append(aLine, pEnd + 2, sizeof(aLine));

		str_append(aLineWithoutIps, pLine, pStart - pLine + 1);
		str_append(aLineWithoutIps, "XXX", sizeof(aLineWithoutIps));
		str_append(aLineWithoutIps, pEnd + 2, sizeof(aLineWithoutIps));

		pLine = aLine;
		pLineWithoutIps = aLineWithoutIps;
	}

	if(ClientID == -1)
	{
		for(int i = 0; i < MAX_CLIENTS; i++)
		{
			if(m_aClients[i].m_State != CClient::STATE_EMPTY && m_aClients[i].m_Authed >= AUTHED_ADMIN)
				SendRconLine(i, m_aClients[i].m_ShowIps ? pLine : pLineWithoutIps);
		}
	}
	else
	{
		if(m_aClients[ClientID].m_State != CClient::STATE_EMPTY)
			SendRconLine(ClientID, m_aClients[ClientID].m_ShowIps ? pLine : pLineWithoutIps);
	}
}

void CServer::SendRconCmdAdd(const IConsole::CCommandInfo *pCommandInfo, int ClientID)
{
	CMsgPacker Msg(NETMSG_RCON_CMD_ADD, true);
	Msg.AddString(pCommandInfo->m_pName, IConsole::TEMPCMD_NAME_LENGTH);
	Msg.AddString(pCommandInfo->m_pHelp, IConsole::TEMPCMD_HELP_LENGTH);
	Msg.AddString(pCommandInfo->m_pParams, IConsole::TEMPCMD_PARAMS_LENGTH);
	SendMsg(&Msg, MSGFLAG_VITAL, ClientID);
}

void CServer::SendRconCmdRem(const IConsole::CCommandInfo *pCommandInfo, int ClientID)
{
	CMsgPacker Msg(NETMSG_RCON_CMD_REM, true);
	Msg.AddString(pCommandInfo->m_pName, 256);
	SendMsg(&Msg, MSGFLAG_VITAL, ClientID);
}

void CServer::UpdateClientRconCommands()
{
	int ClientID = Tick() % MAX_CLIENTS;

	if(m_aClients[ClientID].m_State != CClient::STATE_EMPTY && m_aClients[ClientID].m_Authed)
	{
		int ConsoleAccessLevel = m_aClients[ClientID].m_Authed == AUTHED_ADMIN ? IConsole::ACCESS_LEVEL_ADMIN : m_aClients[ClientID].m_Authed == AUTHED_MOD ? IConsole::ACCESS_LEVEL_MOD : IConsole::ACCESS_LEVEL_HELPER;
		for(int i = 0; i < MAX_RCONCMD_SEND && m_aClients[ClientID].m_pRconCmdToSend; ++i)
		{
			SendRconCmdAdd(m_aClients[ClientID].m_pRconCmdToSend, ClientID);
			m_aClients[ClientID].m_pRconCmdToSend = m_aClients[ClientID].m_pRconCmdToSend->NextCommandInfo(ConsoleAccessLevel, CFGFLAG_SERVER);
		}
	}
}

static inline int MsgFromSixup(int Msg, bool System)
{
	if(System)
	{
		if(Msg == NETMSG_INFO)
			;
		else if(Msg >= 14 && Msg <= 15)
			Msg += 11;
		else if(Msg >= 18 && Msg <= 28)
			Msg = NETMSG_READY + Msg - 18;
		else if(Msg < OFFSET_UUID)
			return -1;
	}

	return Msg;
}

void CServer::ProcessClientPacket(CNetChunk *pPacket)
{
	int ClientID = pPacket->m_ClientID;
	CUnpacker Unpacker;
	Unpacker.Reset(pPacket->m_pData, pPacket->m_DataSize);
	CMsgPacker Packer(NETMSG_EX, true);

	// unpack msgid and system flag
	int Msg;
	bool Sys;
	CUuid Uuid;

	int Result = UnpackMessageID(&Msg, &Sys, &Uuid, &Unpacker, &Packer);
	if(Result == UNPACKMESSAGE_ERROR)
	{
		return;
	}

	if(m_aClients[ClientID].m_Sixup && (Msg = MsgFromSixup(Msg, Sys)) < 0)
	{
		return;
	}

	if(Config()->m_SvNetlimit && Msg != NETMSG_REQUEST_MAP_DATA)
	{
		int64_t Now = time_get();
		int64_t Diff = Now - m_aClients[ClientID].m_TrafficSince;
		double Alpha = Config()->m_SvNetlimitAlpha / 100.0;
		double Limit = (double)(Config()->m_SvNetlimit * 1024) / time_freq();

		if(m_aClients[ClientID].m_Traffic > Limit)
		{
			m_NetServer.NetBan()->BanAddr(&pPacket->m_Address, 600, "Stressing network");
			return;
		}
		if(Diff > 100)
		{
			m_aClients[ClientID].m_Traffic = (Alpha * ((double)pPacket->m_DataSize / Diff)) + (1.0 - Alpha) * m_aClients[ClientID].m_Traffic;
			m_aClients[ClientID].m_TrafficSince = Now;
		}
	}

	if(Result == UNPACKMESSAGE_ANSWER)
	{
		SendMsg(&Packer, MSGFLAG_VITAL, ClientID);
	}

	if(Sys)
	{
		// system message
		if(Msg == NETMSG_CLIENTVER)
		{
			if((pPacket->m_Flags & NET_CHUNKFLAG_VITAL) != 0 && m_aClients[ClientID].m_State == CClient::STATE_PREAUTH)
			{
				CUuid *pConnectionID = (CUuid *)Unpacker.GetRaw(sizeof(*pConnectionID));
				int DDNetVersion = Unpacker.GetInt();
				const char *pDDNetVersionStr = Unpacker.GetString(CUnpacker::SANITIZE_CC);
				if(Unpacker.Error() || !str_utf8_check(pDDNetVersionStr) || DDNetVersion < 0)
				{
					return;
				}
				m_aClients[ClientID].m_ConnectionID = *pConnectionID;
				m_aClients[ClientID].m_DDNetVersion = DDNetVersion;
				str_copy(m_aClients[ClientID].m_aDDNetVersionStr, pDDNetVersionStr);
				m_aClients[ClientID].m_DDNetVersionSettled = true;
				m_aClients[ClientID].m_GotDDNetVersionPacket = true;
				m_aClients[ClientID].m_State = CClient::STATE_AUTH;
			}
		}
		else if(Msg == NETMSG_INFO)
		{
			if((pPacket->m_Flags & NET_CHUNKFLAG_VITAL) != 0 && (m_aClients[ClientID].m_State == CClient::STATE_PREAUTH || m_aClients[ClientID].m_State == CClient::STATE_AUTH))
			{
				const char *pVersion = Unpacker.GetString(CUnpacker::SANITIZE_CC);
				if(!str_utf8_check(pVersion))
				{
					return;
				}
				if(str_comp(pVersion, GameServer()->NetVersion()) != 0 && str_comp(pVersion, "0.7 802f1be60a05665f") != 0)
				{
					// wrong version
					char aReason[256];
					str_format(aReason, sizeof(aReason), "Wrong version. Server is running '%s' and client '%s'", GameServer()->NetVersion(), pVersion);
					m_NetServer.Drop(ClientID, aReason);
					return;
				}

				const char *pPassword = Unpacker.GetString(CUnpacker::SANITIZE_CC);
				if(!str_utf8_check(pPassword))
				{
					return;
				}
				if(Config()->m_Password[0] != 0 && str_comp(Config()->m_Password, pPassword) != 0)
				{
					// wrong password
					m_NetServer.Drop(ClientID, "Wrong password");
					return;
				}

				// reserved slot
				if(ClientID >= (Config()->m_SvMaxClients - Config()->m_SvReservedSlots) && Config()->m_SvReservedSlotsPass[0] != 0 && str_comp(Config()->m_SvReservedSlotsPass, pPassword) != 0)
				{
					m_NetServer.Drop(ClientID, "This server is full");
					return;
				}

				m_aClients[ClientID].m_State = CClient::STATE_CONNECTING;
				SendRconType(ClientID, m_AuthManager.NumNonDefaultKeys() > 0);
				SendCapabilities(ClientID);
				SendMap(ClientID);
			}
		}
		else if(Msg == NETMSG_REQUEST_MAP_DATA)
		{
			if((pPacket->m_Flags & NET_CHUNKFLAG_VITAL) == 0 || m_aClients[ClientID].m_State < CClient::STATE_CONNECTING)
				return;

			if(m_aClients[ClientID].m_Sixup)
			{
				for(int i = 0; i < Config()->m_SvMapWindow; i++)
				{
					SendMapData(ClientID, m_aClients[ClientID].m_NextMapChunk++);
				}
				return;
			}

			int Chunk = Unpacker.GetInt();
			if(Chunk != m_aClients[ClientID].m_NextMapChunk || !Config()->m_SvFastDownload)
			{
				SendMapData(ClientID, Chunk);
				return;
			}

			if(Chunk == 0)
			{
				for(int i = 0; i < Config()->m_SvMapWindow; i++)
				{
					SendMapData(ClientID, i);
				}
			}
			SendMapData(ClientID, Config()->m_SvMapWindow + m_aClients[ClientID].m_NextMapChunk);
			m_aClients[ClientID].m_NextMapChunk++;
		}
		else if(Msg == NETMSG_READY)
		{
			if((pPacket->m_Flags & NET_CHUNKFLAG_VITAL) != 0 && (m_aClients[ClientID].m_State == CClient::STATE_CONNECTING))
			{
				char aAddrStr[NETADDR_MAXSTRSIZE];
				net_addr_str(m_NetServer.ClientAddr(ClientID), aAddrStr, sizeof(aAddrStr), true);

				char aBuf[256];
				str_format(aBuf, sizeof(aBuf), "player is ready. ClientID=%d addr=<{%s}> secure=%s", ClientID, aAddrStr, m_NetServer.HasSecurityToken(ClientID) ? "yes" : "no");
				Console()->Print(IConsole::OUTPUT_LEVEL_ADDINFO, "server", aBuf);

				void *pPersistentData = 0;
				if(m_aClients[ClientID].m_HasPersistentData)
				{
					pPersistentData = m_aClients[ClientID].m_pPersistentData;
					m_aClients[ClientID].m_HasPersistentData = false;
				}
				m_aClients[ClientID].m_State = CClient::STATE_READY;
				GameServer()->OnClientConnected(ClientID, pPersistentData);
			}

			SendConnectionReady(ClientID);
		}
		else if(Msg == NETMSG_ENTERGAME)
		{
			if((pPacket->m_Flags & NET_CHUNKFLAG_VITAL) != 0 && m_aClients[ClientID].m_State == CClient::STATE_READY && GameServer()->IsClientReady(ClientID))
			{
				char aAddrStr[NETADDR_MAXSTRSIZE];
				net_addr_str(m_NetServer.ClientAddr(ClientID), aAddrStr, sizeof(aAddrStr), true);

				char aBuf[256];
				str_format(aBuf, sizeof(aBuf), "player has entered the game. ClientID=%d addr=<{%s}> sixup=%d", ClientID, aAddrStr, IsSixup(ClientID));
				Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", aBuf);
				m_aClients[ClientID].m_State = CClient::STATE_INGAME;
				if(!IsSixup(ClientID))
				{
					SendServerInfo(m_NetServer.ClientAddr(ClientID), -1, SERVERINFO_EXTENDED, false);
				}
				else
				{
					CMsgPacker Msgp(4, true, true); //NETMSG_SERVERINFO //TODO: Import the shared protocol from 7 as well
					GetServerInfoSixup(&Msgp, -1, false);
					SendMsg(&Msgp, MSGFLAG_VITAL | MSGFLAG_FLUSH, ClientID);
				}
				GameServer()->OnClientEnter(ClientID);
			}
		}
		else if(Msg == NETMSG_INPUT)
		{
			CClient::CInput *pInput;
			int64_t TagTime;

			m_aClients[ClientID].m_LastAckedSnapshot = Unpacker.GetInt();
			int IntendedTick = Unpacker.GetInt();
			int Size = Unpacker.GetInt();

			// check for errors
			if(Unpacker.Error() || Size / 4 > MAX_INPUT_SIZE)
				return;

			if(m_aClients[ClientID].m_LastAckedSnapshot > 0)
				m_aClients[ClientID].m_SnapRate = CClient::SNAPRATE_FULL;

			if(m_aClients[ClientID].m_Snapshots.Get(m_aClients[ClientID].m_LastAckedSnapshot, &TagTime, 0, 0) >= 0)
				m_aClients[ClientID].m_Latency = (int)(((time_get() - TagTime) * 1000) / time_freq());

			// add message to report the input timing
			// skip packets that are old
			if(IntendedTick > m_aClients[ClientID].m_LastInputTick)
			{
				int TimeLeft = ((TickStartTime(IntendedTick) - time_get()) * 1000) / time_freq();

				CMsgPacker Msgp(NETMSG_INPUTTIMING, true);
				Msgp.AddInt(IntendedTick);
				Msgp.AddInt(TimeLeft);
				SendMsg(&Msgp, 0, ClientID);
			}

			m_aClients[ClientID].m_LastInputTick = IntendedTick;

			pInput = &m_aClients[ClientID].m_aInputs[m_aClients[ClientID].m_CurrentInput];

			if(IntendedTick <= Tick())
				IntendedTick = Tick() + 1;

			pInput->m_GameTick = IntendedTick;

			for(int i = 0; i < Size / 4; i++)
				pInput->m_aData[i] = Unpacker.GetInt();

			GameServer()->OnClientPrepareInput(ClientID, pInput->m_aData);
			mem_copy(m_aClients[ClientID].m_LatestInput.m_aData, pInput->m_aData, MAX_INPUT_SIZE * sizeof(int));

			m_aClients[ClientID].m_CurrentInput++;
			m_aClients[ClientID].m_CurrentInput %= 200;

			// call the mod with the fresh input data
			if(m_aClients[ClientID].m_State == CClient::STATE_INGAME)
				GameServer()->OnClientDirectInput(ClientID, m_aClients[ClientID].m_LatestInput.m_aData);
		}
		else if(Msg == NETMSG_RCON_CMD)
		{
			const char *pCmd = Unpacker.GetString();
			if(!str_utf8_check(pCmd))
			{
				return;
			}
			if(Unpacker.Error() == 0 && !str_comp(pCmd, "crashmeplx"))
			{
				int Version = m_aClients[ClientID].m_DDNetVersion;
				if(GameServer()->PlayerExists(ClientID) && Version < VERSION_DDNET_OLD)
				{
					m_aClients[ClientID].m_DDNetVersion = VERSION_DDNET_OLD;
				}
			}
			else if((pPacket->m_Flags & NET_CHUNKFLAG_VITAL) != 0 && Unpacker.Error() == 0 && m_aClients[ClientID].m_Authed)
			{
				if(GameServer()->PlayerExists(ClientID))
				{
					char aBuf[256];
					str_format(aBuf, sizeof(aBuf), "ClientID=%d rcon='%s'", ClientID, pCmd);
					Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", aBuf);
					m_RconClientID = ClientID;
					m_RconAuthLevel = m_aClients[ClientID].m_Authed;
					Console()->SetAccessLevel(m_aClients[ClientID].m_Authed == AUTHED_ADMIN ? IConsole::ACCESS_LEVEL_ADMIN : m_aClients[ClientID].m_Authed == AUTHED_MOD ? IConsole::ACCESS_LEVEL_MOD : m_aClients[ClientID].m_Authed == AUTHED_HELPER ? IConsole::ACCESS_LEVEL_HELPER : IConsole::ACCESS_LEVEL_USER);
					{
						CRconClientLogger Logger(this, ClientID);
						CLogScope Scope(&Logger);
						Console()->ExecuteLineFlag(pCmd, CFGFLAG_SERVER, ClientID);
					}
					Console()->SetAccessLevel(IConsole::ACCESS_LEVEL_ADMIN);
					m_RconClientID = IServer::RCON_CID_SERV;
					m_RconAuthLevel = AUTHED_ADMIN;
				}
			}
		}
		else if(Msg == NETMSG_RCON_AUTH)
		{
			const char *pName = "";
			if(!IsSixup(ClientID))
				pName = Unpacker.GetString(CUnpacker::SANITIZE_CC); // login name, now used
			const char *pPw = Unpacker.GetString(CUnpacker::SANITIZE_CC);
			if(!str_utf8_check(pPw) || !str_utf8_check(pName))
			{
				return;
			}

			if((pPacket->m_Flags & NET_CHUNKFLAG_VITAL) != 0 && Unpacker.Error() == 0)
			{
				int AuthLevel = -1;
				int KeySlot = -1;

				if(!pName[0])
				{
					if(m_AuthManager.CheckKey((KeySlot = m_AuthManager.DefaultKey(AUTHED_ADMIN)), pPw))
						AuthLevel = AUTHED_ADMIN;
					else if(m_AuthManager.CheckKey((KeySlot = m_AuthManager.DefaultKey(AUTHED_MOD)), pPw))
						AuthLevel = AUTHED_MOD;
					else if(m_AuthManager.CheckKey((KeySlot = m_AuthManager.DefaultKey(AUTHED_HELPER)), pPw))
						AuthLevel = AUTHED_HELPER;
				}
				else
				{
					KeySlot = m_AuthManager.FindKey(pName);
					if(m_AuthManager.CheckKey(KeySlot, pPw))
						AuthLevel = m_AuthManager.KeyLevel(KeySlot);
				}

				if(AuthLevel != -1)
				{
					if(m_aClients[ClientID].m_Authed != AuthLevel)
					{
						if(!IsSixup(ClientID))
						{
							CMsgPacker Msgp(NETMSG_RCON_AUTH_STATUS, true);
							Msgp.AddInt(1); //authed
							Msgp.AddInt(1); //cmdlist
							SendMsg(&Msgp, MSGFLAG_VITAL, ClientID);
						}
						else
						{
							CMsgPacker Msgp(11, true, true); //NETMSG_RCON_AUTH_ON
							SendMsg(&Msgp, MSGFLAG_VITAL, ClientID);
						}

						m_aClients[ClientID].m_Authed = AuthLevel; // Keeping m_Authed around is unwise...
						m_aClients[ClientID].m_AuthKey = KeySlot;
						int SendRconCmds = IsSixup(ClientID) ? true : Unpacker.GetInt();
						if(Unpacker.Error() == 0 && SendRconCmds)
							// AUTHED_ADMIN - AuthLevel gets the proper IConsole::ACCESS_LEVEL_<x>
							m_aClients[ClientID].m_pRconCmdToSend = Console()->FirstCommandInfo(AUTHED_ADMIN - AuthLevel, CFGFLAG_SERVER);

						char aBuf[256];
						const char *pIdent = m_AuthManager.KeyIdent(KeySlot);
						switch(AuthLevel)
						{
						case AUTHED_ADMIN:
						{
							SendRconLine(ClientID, "Admin authentication successful. Full remote console access granted.");
							str_format(aBuf, sizeof(aBuf), "ClientID=%d authed with key=%s (admin)", ClientID, pIdent);
							break;
						}
						case AUTHED_MOD:
						{
							SendRconLine(ClientID, "Moderator authentication successful. Limited remote console access granted.");
							str_format(aBuf, sizeof(aBuf), "ClientID=%d authed with key=%s (moderator)", ClientID, pIdent);
							break;
						}
						case AUTHED_HELPER:
						{
							SendRconLine(ClientID, "Helper authentication successful. Limited remote console access granted.");
							str_format(aBuf, sizeof(aBuf), "ClientID=%d authed with key=%s (helper)", ClientID, pIdent);
							break;
						}
						}
						Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", aBuf);

						// DDRace
						GameServer()->OnSetAuthed(ClientID, AuthLevel);
					}
				}
				else if(Config()->m_SvRconMaxTries)
				{
					m_aClients[ClientID].m_AuthTries++;
					char aBuf[128];
					str_format(aBuf, sizeof(aBuf), "Wrong password %d/%d.", m_aClients[ClientID].m_AuthTries, Config()->m_SvRconMaxTries);
					SendRconLine(ClientID, aBuf);
					if(m_aClients[ClientID].m_AuthTries >= Config()->m_SvRconMaxTries)
					{
						if(!Config()->m_SvRconBantime)
							m_NetServer.Drop(ClientID, "Too many remote console authentication tries");
						else
							m_ServerBan.BanAddr(m_NetServer.ClientAddr(ClientID), Config()->m_SvRconBantime * 60, "Too many remote console authentication tries");
					}
				}
				else
				{
					SendRconLine(ClientID, "Wrong password.");
				}
			}
		}
		else if(Msg == NETMSG_PING)
		{
			CMsgPacker Msgp(NETMSG_PING_REPLY, true);
			SendMsg(&Msgp, 0, ClientID);
		}
		else if(Msg == NETMSG_PINGEX)
		{
			CUuid *pID = (CUuid *)Unpacker.GetRaw(sizeof(*pID));
			if(Unpacker.Error())
			{
				return;
			}
			CMsgPacker Msgp(NETMSG_PONGEX, true);
			Msgp.AddRaw(pID, sizeof(*pID));
			SendMsg(&Msgp, MSGFLAG_FLUSH, ClientID);
		}
		else
		{
			if(Config()->m_Debug)
			{
				constexpr int MaxDumpedDataSize = 32;
				char aBuf[MaxDumpedDataSize * 3 + 1];
				str_hex(aBuf, sizeof(aBuf), pPacket->m_pData, minimum(pPacket->m_DataSize, MaxDumpedDataSize));

				char aBufMsg[256];
				str_format(aBufMsg, sizeof(aBufMsg), "strange message ClientID=%d msg=%d data_size=%d", ClientID, Msg, pPacket->m_DataSize);
				Console()->Print(IConsole::OUTPUT_LEVEL_DEBUG, "server", aBufMsg);
				Console()->Print(IConsole::OUTPUT_LEVEL_DEBUG, "server", aBuf);
			}
		}
	}
	else
	{
		// game message
		if((pPacket->m_Flags & NET_CHUNKFLAG_VITAL) != 0 && m_aClients[ClientID].m_State >= CClient::STATE_READY)
			GameServer()->OnMessage(Msg, &Unpacker, ClientID);
	}
}

bool CServer::RateLimitServerInfoConnless()
{
	bool SendClients = true;
	if(Config()->m_SvServerInfoPerSecond)
	{
		SendClients = m_ServerInfoNumRequests <= Config()->m_SvServerInfoPerSecond;
		const int64_t Now = Tick();

		if(Now <= m_ServerInfoFirstRequest + TickSpeed())
		{
			m_ServerInfoNumRequests++;
		}
		else
		{
			m_ServerInfoNumRequests = 1;
			m_ServerInfoFirstRequest = Now;
		}
	}

	return SendClients;
}

void CServer::SendServerInfoConnless(const NETADDR *pAddr, int Token, int Type)
{
	SendServerInfo(pAddr, Token, Type, RateLimitServerInfoConnless());
}

static inline int GetCacheIndex(int Type, bool SendClient)
{
	if(Type == SERVERINFO_INGAME)
		Type = SERVERINFO_VANILLA;
	else if(Type == SERVERINFO_EXTENDED_MORE)
		Type = SERVERINFO_EXTENDED;

	return Type * 2 + SendClient;
}

CServer::CCache::CCache()
{
	m_Cache.clear();
}

CServer::CCache::~CCache()
{
	Clear();
}

CServer::CCache::CCacheChunk::CCacheChunk(const void *pData, int Size)
{
	m_vData.assign((const uint8_t *)pData, (const uint8_t *)pData + Size);
}

void CServer::CCache::AddChunk(const void *pData, int Size)
{
	m_Cache.emplace_back(pData, Size);
}

void CServer::CCache::Clear()
{
	m_Cache.clear();
}

void CServer::CacheServerInfo(CCache *pCache, int Type, bool SendClients)
{
	pCache->Clear();

	// One chance to improve the protocol!
	CPacker p;
	char aBuf[128];

	// count the players
	int PlayerCount = 0, ClientCount = 0;
	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(m_aClients[i].m_State != CClient::STATE_EMPTY)
		{
			if(GameServer()->IsClientPlayer(i))
				PlayerCount++;

			ClientCount++;
		}
	}

	p.Reset();

#define ADD_RAW(p, x) (p).AddRaw(x, sizeof(x))
#define ADD_INT(p, x) \
	do \
	{ \
		str_format(aBuf, sizeof(aBuf), "%d", x); \
		(p).AddString(aBuf, 0); \
	} while(0)

	p.AddString(GameServer()->Version(), 32);
	if(Type != SERVERINFO_VANILLA)
	{
		p.AddString(Config()->m_SvName, 256);
	}
	else
	{
		if(m_NetServer.MaxClients() <= VANILLA_MAX_CLIENTS)
		{
			p.AddString(Config()->m_SvName, 64);
		}
		else
		{
			const int MaxClients = maximum(ClientCount, m_NetServer.MaxClients() - Config()->m_SvReservedSlots);
			str_format(aBuf, sizeof(aBuf), "%s [%d/%d]", Config()->m_SvName, ClientCount, MaxClients);
			p.AddString(aBuf, 64);
		}
	}
	p.AddString(GetMapName(), 32);

	if(Type == SERVERINFO_EXTENDED)
	{
		ADD_INT(p, m_aCurrentMapCrc[MAP_TYPE_SIX]);
		ADD_INT(p, m_aCurrentMapSize[MAP_TYPE_SIX]);
	}

	// gametype
	p.AddString(GameServer()->GameType(), 16);

	// flags
	ADD_INT(p, Config()->m_Password[0] ? SERVER_FLAG_PASSWORD : 0);

	int MaxClients = m_NetServer.MaxClients();
	// How many clients the used serverinfo protocol supports, has to be tracked
	// separately to make sure we don't subtract the reserved slots from it
	int MaxClientsProtocol = MAX_CLIENTS;
	if(Type == SERVERINFO_VANILLA || Type == SERVERINFO_INGAME)
	{
		if(ClientCount >= VANILLA_MAX_CLIENTS)
		{
			if(ClientCount < MaxClients)
				ClientCount = VANILLA_MAX_CLIENTS - 1;
			else
				ClientCount = VANILLA_MAX_CLIENTS;
		}
		MaxClientsProtocol = VANILLA_MAX_CLIENTS;
		if(PlayerCount > ClientCount)
			PlayerCount = ClientCount;
	}

	ADD_INT(p, PlayerCount); // num players
	ADD_INT(p, minimum(MaxClientsProtocol, maximum(MaxClients - maximum(Config()->m_SvSpectatorSlots, Config()->m_SvReservedSlots), PlayerCount))); // max players
	ADD_INT(p, ClientCount); // num clients
	ADD_INT(p, minimum(MaxClientsProtocol, maximum(MaxClients - Config()->m_SvReservedSlots, ClientCount))); // max clients

	if(Type == SERVERINFO_EXTENDED)
		p.AddString("", 0); // extra info, reserved

	const void *pPrefix = p.Data();
	int PrefixSize = p.Size();

	CPacker q;
	int ChunksStored = 0;
	int PlayersStored = 0;

#define SAVE(size) \
	do \
	{ \
		pCache->AddChunk(q.Data(), size); \
		ChunksStored++; \
	} while(0)

#define RESET() \
	do \
	{ \
		q.Reset(); \
		q.AddRaw(pPrefix, PrefixSize); \
	} while(0)

	RESET();

	if(Type == SERVERINFO_64_LEGACY)
		q.AddInt(PlayersStored); // offset

	if(!SendClients)
	{
		SAVE(q.Size());
		return;
	}

	if(Type == SERVERINFO_EXTENDED)
	{
		pPrefix = "";
		PrefixSize = 0;
	}

	int Remaining;
	switch(Type)
	{
	case SERVERINFO_EXTENDED: Remaining = -1; break;
	case SERVERINFO_64_LEGACY: Remaining = 24; break;
	case SERVERINFO_VANILLA: Remaining = VANILLA_MAX_CLIENTS; break;
	case SERVERINFO_INGAME: Remaining = VANILLA_MAX_CLIENTS; break;
	default: dbg_assert(0, "caught earlier, unreachable"); return;
	}

	// Use the following strategy for sending:
	// For vanilla, send the first 16 players.
	// For legacy 64p, send 24 players per packet.
	// For extended, send as much players as possible.

	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(m_aClients[i].m_State != CClient::STATE_EMPTY)
		{
			if(Remaining == 0)
			{
				if(Type == SERVERINFO_VANILLA || Type == SERVERINFO_INGAME)
					break;

				// Otherwise we're SERVERINFO_64_LEGACY.
				SAVE(q.Size());
				RESET();
				q.AddInt(PlayersStored); // offset
				Remaining = 24;
			}
			if(Remaining > 0)
			{
				Remaining--;
			}

			int PreviousSize = q.Size();

			q.AddString(ClientName(i), MAX_NAME_LENGTH); // client name
			q.AddString(ClientClan(i), MAX_CLAN_LENGTH); // client clan

			ADD_INT(q, m_aClients[i].m_Country); // client country
			ADD_INT(q, m_aClients[i].m_Score); // client score
			ADD_INT(q, GameServer()->IsClientPlayer(i) ? 1 : 0); // is player?
			if(Type == SERVERINFO_EXTENDED)
				q.AddString("", 0); // extra info, reserved

			if(Type == SERVERINFO_EXTENDED)
			{
				if(q.Size() >= NET_MAX_PAYLOAD - 18) // 8 bytes for type, 10 bytes for the largest token
				{
					// Retry current player.
					i--;
					SAVE(PreviousSize);
					RESET();
					ADD_INT(q, ChunksStored);
					q.AddString("", 0); // extra info, reserved
					continue;
				}
			}
			PlayersStored++;
		}
	}

	SAVE(q.Size());
#undef SAVE
#undef RESET
#undef ADD_RAW
#undef ADD_INT
}

void CServer::CacheServerInfoSixup(CCache *pCache, bool SendClients)
{
	pCache->Clear();

	CPacker Packer;
	Packer.Reset();

	// Could be moved to a separate function and cached
	// count the players
	int PlayerCount = 0, ClientCount = 0;
	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(m_aClients[i].m_State != CClient::STATE_EMPTY)
		{
			if(GameServer()->IsClientPlayer(i))
				PlayerCount++;

			ClientCount++;
		}
	}

	char aVersion[32];
	str_format(aVersion, sizeof(aVersion), "0.7↔%s", GameServer()->Version());
	Packer.AddString(aVersion, 32);
	Packer.AddString(Config()->m_SvName, 64);
	Packer.AddString(Config()->m_SvHostname, 128);
	Packer.AddString(GetMapName(), 32);

	// gametype
	Packer.AddString(GameServer()->GameType(), 16);

	// flags
	int Flags = SERVER_FLAG_TIMESCORE;
	if(Config()->m_Password[0]) // password set
		Flags |= SERVER_FLAG_PASSWORD;
	Packer.AddInt(Flags);

	int MaxClients = m_NetServer.MaxClients();
	Packer.AddInt(Config()->m_SvSkillLevel); // server skill level
	Packer.AddInt(PlayerCount); // num players
	Packer.AddInt(maximum(MaxClients - maximum(Config()->m_SvSpectatorSlots, Config()->m_SvReservedSlots), PlayerCount)); // max players
	Packer.AddInt(ClientCount); // num clients
	Packer.AddInt(maximum(MaxClients - Config()->m_SvReservedSlots, ClientCount)); // max clients

	if(SendClients)
	{
		for(int i = 0; i < MAX_CLIENTS; i++)
		{
			if(m_aClients[i].m_State != CClient::STATE_EMPTY)
			{
				Packer.AddString(ClientName(i), MAX_NAME_LENGTH); // client name
				Packer.AddString(ClientClan(i), MAX_CLAN_LENGTH); // client clan
				Packer.AddInt(m_aClients[i].m_Country); // client country
				Packer.AddInt(m_aClients[i].m_Score == -9999 ? -1 : -m_aClients[i].m_Score); // client score
				Packer.AddInt(GameServer()->IsClientPlayer(i) ? 0 : 1); // flag spectator=1, bot=2 (player=0)
			}
		}
	}

	pCache->AddChunk(Packer.Data(), Packer.Size());
}

void CServer::SendServerInfo(const NETADDR *pAddr, int Token, int Type, bool SendClients)
{
	CPacker p;
	char aBuf[128];
	p.Reset();

	CCache *pCache = &m_aServerInfoCache[GetCacheIndex(Type, SendClients)];

#define ADD_RAW(p, x) (p).AddRaw(x, sizeof(x))
#define ADD_INT(p, x) \
	do \
	{ \
		str_format(aBuf, sizeof(aBuf), "%d", x); \
		(p).AddString(aBuf, 0); \
	} while(0)

	CNetChunk Packet;
	Packet.m_ClientID = -1;
	Packet.m_Address = *pAddr;
	Packet.m_Flags = NETSENDFLAG_CONNLESS;

	for(const auto &Chunk : pCache->m_Cache)
	{
		p.Reset();
		if(Type == SERVERINFO_EXTENDED)
		{
			if(&Chunk == &pCache->m_Cache.front())
				p.AddRaw(SERVERBROWSE_INFO_EXTENDED, sizeof(SERVERBROWSE_INFO_EXTENDED));
			else
				p.AddRaw(SERVERBROWSE_INFO_EXTENDED_MORE, sizeof(SERVERBROWSE_INFO_EXTENDED_MORE));
			ADD_INT(p, Token);
		}
		else if(Type == SERVERINFO_64_LEGACY)
		{
			ADD_RAW(p, SERVERBROWSE_INFO_64_LEGACY);
			ADD_INT(p, Token);
		}
		else if(Type == SERVERINFO_VANILLA || Type == SERVERINFO_INGAME)
		{
			ADD_RAW(p, SERVERBROWSE_INFO);
			ADD_INT(p, Token);
		}
		else
		{
			dbg_assert(false, "unknown serverinfo type");
		}

		p.AddRaw(Chunk.m_vData.data(), Chunk.m_vData.size());
		Packet.m_pData = p.Data();
		Packet.m_DataSize = p.Size();
		m_NetServer.Send(&Packet);
	}
}

void CServer::GetServerInfoSixup(CPacker *pPacker, int Token, bool SendClients)
{
	if(Token != -1)
	{
		pPacker->Reset();
		pPacker->AddRaw(SERVERBROWSE_INFO, sizeof(SERVERBROWSE_INFO));
		pPacker->AddInt(Token);
	}

	SendClients = SendClients && Token != -1;

	CCache::CCacheChunk &FirstChunk = m_aSixupServerInfoCache[SendClients].m_Cache.front();
	pPacker->AddRaw(FirstChunk.m_vData.data(), FirstChunk.m_vData.size());
}

void CServer::ExpireServerInfo()
{
	m_ServerInfoNeedsUpdate = true;
}

void CServer::UpdateRegisterServerInfo()
{
	// count the players
	int PlayerCount = 0, ClientCount = 0;
	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(m_aClients[i].m_State != CClient::STATE_EMPTY)
		{
			if(GameServer()->IsClientPlayer(i))
				PlayerCount++;

			ClientCount++;
		}
	}

	int MaxPlayers = maximum(m_NetServer.MaxClients() - maximum(g_Config.m_SvSpectatorSlots, g_Config.m_SvReservedSlots), PlayerCount);
	int MaxClients = maximum(m_NetServer.MaxClients() - g_Config.m_SvReservedSlots, ClientCount);
	char aName[256];
	char aGameType[32];
	char aMapName[64];
	char aVersion[64];
	char aMapSha256[SHA256_MAXSTRSIZE];

	sha256_str(m_aCurrentMapSha256[MAP_TYPE_SIX], aMapSha256, sizeof(aMapSha256));

	char aInfo[16384];
	str_format(aInfo, sizeof(aInfo),
		"{"
		"\"max_clients\":%d,"
		"\"max_players\":%d,"
		"\"passworded\":%s,"
		"\"game_type\":\"%s\","
		"\"name\":\"%s\","
		"\"map\":{"
		"\"name\":\"%s\","
		"\"sha256\":\"%s\","
		"\"size\":%d"
		"},"
		"\"version\":\"%s\","
		"\"clients\":[",
		MaxClients,
		MaxPlayers,
		JsonBool(g_Config.m_Password[0]),
		EscapeJson(aGameType, sizeof(aGameType), GameServer()->GameType()),
		EscapeJson(aName, sizeof(aName), g_Config.m_SvName),
		EscapeJson(aMapName, sizeof(aMapName), m_aCurrentMap),
		aMapSha256,
		m_aCurrentMapSize[MAP_TYPE_SIX],
		EscapeJson(aVersion, sizeof(aVersion), GameServer()->Version()));

	bool FirstPlayer = true;
	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(m_aClients[i].m_State != CClient::STATE_EMPTY)
		{
			char aCName[32];
			char aCClan[32];

			char aExtraPlayerInfo[512];
			GameServer()->OnUpdatePlayerServerInfo(aExtraPlayerInfo, sizeof(aExtraPlayerInfo), i);

			char aClientInfo[1024];
			str_format(aClientInfo, sizeof(aClientInfo),
				"%s{"
				"\"name\":\"%s\","
				"\"clan\":\"%s\","
				"\"country\":%d,"
				"\"score\":%d,"
				"\"is_player\":%s"
				"%s"
				"}",
				!FirstPlayer ? "," : "",
				EscapeJson(aCName, sizeof(aCName), ClientName(i)),
				EscapeJson(aCClan, sizeof(aCClan), ClientClan(i)),
				m_aClients[i].m_Country,
				m_aClients[i].m_Score,
				JsonBool(GameServer()->IsClientPlayer(i)),
				aExtraPlayerInfo);
			str_append(aInfo, aClientInfo, sizeof(aInfo));
			FirstPlayer = false;
		}
	}

	str_append(aInfo, "]}", sizeof(aInfo));

	m_pRegister->OnNewInfo(aInfo);
}

void CServer::UpdateServerInfo(bool Resend)
{
	if(m_RunServer == UNINITIALIZED)
		return;

	UpdateRegisterServerInfo();

	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 2; j++)
			CacheServerInfo(&m_aServerInfoCache[i * 2 + j], i, j);

	for(int i = 0; i < 2; i++)
		CacheServerInfoSixup(&m_aSixupServerInfoCache[i], i);

	if(Resend)
	{
		for(int i = 0; i < MaxClients(); ++i)
		{
			if(m_aClients[i].m_State != CClient::STATE_EMPTY)
			{
				if(!IsSixup(i))
					SendServerInfo(m_NetServer.ClientAddr(i), -1, SERVERINFO_INGAME, false);
				else
				{
					CMsgPacker Msg(4, true, true); //NETMSG_SERVERINFO //TODO: Import the shared protocol from 7 as well
					GetServerInfoSixup(&Msg, -1, false);
					SendMsg(&Msg, MSGFLAG_VITAL | MSGFLAG_FLUSH, i);
				}
			}
		}
	}

	m_ServerInfoNeedsUpdate = false;
}

void CServer::PumpNetwork(bool PacketWaiting)
{
	CNetChunk Packet;
	SECURITY_TOKEN ResponseToken;

	m_NetServer.Update();

	if(PacketWaiting)
	{
		// process packets
		while(m_NetServer.Recv(&Packet, &ResponseToken))
		{
			if(Packet.m_ClientID == -1)
			{
				if(ResponseToken == NET_SECURITY_TOKEN_UNKNOWN && m_pRegister->OnPacket(&Packet))
					continue;

				{
					int ExtraToken = 0;
					int Type = -1;
					if(Packet.m_DataSize >= (int)sizeof(SERVERBROWSE_GETINFO) + 1 &&
						mem_comp(Packet.m_pData, SERVERBROWSE_GETINFO, sizeof(SERVERBROWSE_GETINFO)) == 0)
					{
						if(Packet.m_Flags & NETSENDFLAG_EXTENDED)
						{
							Type = SERVERINFO_EXTENDED;
							ExtraToken = (Packet.m_aExtraData[0] << 8) | Packet.m_aExtraData[1];
						}
						else
							Type = SERVERINFO_VANILLA;
					}
					else if(Packet.m_DataSize >= (int)sizeof(SERVERBROWSE_GETINFO_64_LEGACY) + 1 &&
						mem_comp(Packet.m_pData, SERVERBROWSE_GETINFO_64_LEGACY, sizeof(SERVERBROWSE_GETINFO_64_LEGACY)) == 0)
					{
						Type = SERVERINFO_64_LEGACY;
					}
					if(Type == SERVERINFO_VANILLA && ResponseToken != NET_SECURITY_TOKEN_UNKNOWN && Config()->m_SvSixup)
					{
						CUnpacker Unpacker;
						Unpacker.Reset((unsigned char *)Packet.m_pData + sizeof(SERVERBROWSE_GETINFO), Packet.m_DataSize - sizeof(SERVERBROWSE_GETINFO));
						int SrvBrwsToken = Unpacker.GetInt();
						if(Unpacker.Error())
							continue;

						CPacker Packer;
						CNetChunk Response;

						GetServerInfoSixup(&Packer, SrvBrwsToken, RateLimitServerInfoConnless());

						Response.m_ClientID = -1;
						Response.m_Address = Packet.m_Address;
						Response.m_Flags = NETSENDFLAG_CONNLESS;
						Response.m_pData = Packer.Data();
						Response.m_DataSize = Packer.Size();
						m_NetServer.SendConnlessSixup(&Response, ResponseToken);
					}
					else if(Type != -1)
					{
						int Token = ((unsigned char *)Packet.m_pData)[sizeof(SERVERBROWSE_GETINFO)];
						Token |= ExtraToken << 8;
						SendServerInfoConnless(&Packet.m_Address, Token, Type);
					}
				}
			}
			else
			{
				int GameFlags = 0;
				if(Packet.m_Flags & NET_CHUNKFLAG_VITAL)
				{
					GameFlags |= MSGFLAG_VITAL;
				}
				if(Antibot()->OnEngineClientMessage(Packet.m_ClientID, Packet.m_pData, Packet.m_DataSize, GameFlags))
				{
					continue;
				}

				ProcessClientPacket(&Packet);
			}
		}
	}
	{
		unsigned char aBuffer[NET_MAX_PAYLOAD];
		int Flags;
		mem_zero(&Packet, sizeof(Packet));
		Packet.m_pData = aBuffer;
		while(Antibot()->OnEngineSimulateClientMessage(&Packet.m_ClientID, aBuffer, sizeof(aBuffer), &Packet.m_DataSize, &Flags))
		{
			Packet.m_Flags = 0;
			if(Flags & MSGFLAG_VITAL)
			{
				Packet.m_Flags |= NET_CHUNKFLAG_VITAL;
			}
			ProcessClientPacket(&Packet);
		}
	}

	m_ServerBan.Update();
	m_Econ.Update();
}

const char *CServer::GetMapName() const
{
	// get the name of the map without his path
	const char *pMapShortName = &Config()->m_SvMap[0];
	for(int i = 0; i < str_length(Config()->m_SvMap) - 1; i++)
	{
		if(Config()->m_SvMap[i] == '/' || Config()->m_SvMap[i] == '\\')
			pMapShortName = &Config()->m_SvMap[i + 1];
	}
	return pMapShortName;
}

void CServer::ChangeMap(const char *pMap)
{
	str_copy(Config()->m_SvMap, pMap);
	m_MapReload = str_comp(Config()->m_SvMap, m_aCurrentMap) != 0;
}

int CServer::LoadMap(const char *pMapName)
{
	m_MapReload = false;

	char aBuf[IO_MAX_PATH_LENGTH];
	str_format(aBuf, sizeof(aBuf), "maps/%s.map", pMapName);
	GameServer()->OnMapChange(aBuf, sizeof(aBuf));

	if(!m_pMap->Load(aBuf))
		return 0;

	// stop recording when we change map
	for(int i = 0; i < MAX_CLIENTS + 1; i++)
	{
		if(!m_aDemoRecorder[i].IsRecording())
			continue;

		m_aDemoRecorder[i].Stop();

		// remove tmp demos
		if(i < MAX_CLIENTS)
		{
			char aPath[256];
			str_format(aPath, sizeof(aPath), "demos/%s_%d_%d_tmp.demo", m_aCurrentMap, m_NetServer.Address().port, i);
			Storage()->RemoveFile(aPath, IStorage::TYPE_SAVE);
		}
	}

	// reinit snapshot ids
	m_IDPool.TimeoutIDs();

	// get the crc of the map
	m_aCurrentMapSha256[MAP_TYPE_SIX] = m_pMap->Sha256();
	m_aCurrentMapCrc[MAP_TYPE_SIX] = m_pMap->Crc();
	char aBufMsg[256];
	char aSha256[SHA256_MAXSTRSIZE];
	sha256_str(m_aCurrentMapSha256[MAP_TYPE_SIX], aSha256, sizeof(aSha256));
	str_format(aBufMsg, sizeof(aBufMsg), "%s sha256 is %s", aBuf, aSha256);
	Console()->Print(IConsole::OUTPUT_LEVEL_ADDINFO, "server", aBufMsg);

	str_copy(m_aCurrentMap, pMapName);

	// load complete map into memory for download
	{
		free(m_apCurrentMapData[MAP_TYPE_SIX]);
		void *pData;
		Storage()->ReadFile(aBuf, IStorage::TYPE_ALL, &pData, &m_aCurrentMapSize[MAP_TYPE_SIX]);
		m_apCurrentMapData[MAP_TYPE_SIX] = (unsigned char *)pData;
	}

	// load sixup version of the map
	if(Config()->m_SvSixup)
	{
		str_format(aBuf, sizeof(aBuf), "maps7/%s.map", pMapName);
		void *pData;
		if(!Storage()->ReadFile(aBuf, IStorage::TYPE_ALL, &pData, &m_aCurrentMapSize[MAP_TYPE_SIXUP]))
		{
			Config()->m_SvSixup = 0;
			if(m_pRegister)
			{
				m_pRegister->OnConfigChange();
			}
			str_format(aBufMsg, sizeof(aBufMsg), "couldn't load map %s", aBuf);
			Console()->Print(IConsole::OUTPUT_LEVEL_ADDINFO, "sixup", aBufMsg);
			Console()->Print(IConsole::OUTPUT_LEVEL_ADDINFO, "sixup", "disabling 0.7 compatibility");
		}
		else
		{
			free(m_apCurrentMapData[MAP_TYPE_SIXUP]);
			m_apCurrentMapData[MAP_TYPE_SIXUP] = (unsigned char *)pData;

			m_aCurrentMapSha256[MAP_TYPE_SIXUP] = sha256(m_apCurrentMapData[MAP_TYPE_SIXUP], m_aCurrentMapSize[MAP_TYPE_SIXUP]);
			m_aCurrentMapCrc[MAP_TYPE_SIXUP] = crc32(0, m_apCurrentMapData[MAP_TYPE_SIXUP], m_aCurrentMapSize[MAP_TYPE_SIXUP]);
			sha256_str(m_aCurrentMapSha256[MAP_TYPE_SIXUP], aSha256, sizeof(aSha256));
			str_format(aBufMsg, sizeof(aBufMsg), "%s sha256 is %s", aBuf, aSha256);
			Console()->Print(IConsole::OUTPUT_LEVEL_ADDINFO, "sixup", aBufMsg);
		}
	}
	if(!Config()->m_SvSixup)
	{
		free(m_apCurrentMapData[MAP_TYPE_SIXUP]);
		m_apCurrentMapData[MAP_TYPE_SIXUP] = 0;
	}

	for(int i = 0; i < MAX_CLIENTS; i++)
		m_aPrevStates[i] = m_aClients[i].m_State;

	return 1;
}

unsigned long long GenerateSeed()
{
	return std::chrono::system_clock::now().time_since_epoch().count();
}

CPlayer* CServer::AddBot(const char* Name)
{
	for(int ClientID = MAX_CLIENTS-1; ClientID >= 0; ClientID--)
	{
		if (m_aClients[ClientID].m_State == CClient::STATE_EMPTY)
		{
			//m_aClients[ClientID].m_aName = "Bot";
			memcpy(m_aClients[ClientID].m_aName, Name, strlen(Name)+1);

			m_aClients[ClientID].m_State = CClient::STATE_INGAME;
			m_aClients[ClientID].m_SnapRate = CClient::SNAPRATE_FULL;
			m_aClients[ClientID].m_DDNetVersion = VERSION_DDRACE;
			m_aClients[ClientID].m_pRconCmdToSend = 0;

			GameServer()->OnClientConnected(ClientID, nullptr);
			GameServer()->OnClientEnter(ClientID);

			auto gamecontext = ((CGameContext *)GameServer());
			CPlayer *pPlayer = gamecontext->m_apPlayers[ClientID];
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

vec2 angle_to_coords(int angle)
{
	angle -= 402;
	auto reversed = (float)angle / 256.0f;
	// cout << (float)angle / 256.0f << endl;
	int x = (int)(round(cos(reversed) * 299.0f));
	int y = (int)(round(sin(reversed) * 299.0f));
	//cout << x << endl;
	//cout << y << endl;

	return vec2(x, y);
}

int coords_to_angle(float x, float y)
{
	auto TmpAngle = atan2f(y, x);

	int m_Angle = 0;
	//cout << m_Angle << endl;

	constexpr float pi = 3.1415926535897932384626433f;

	if(TmpAngle < -(pi / 2.0f))
	{
		m_Angle = (int)((TmpAngle + (2.0f * pi)) * 256.0f);
		//cout << m_Angle << endl;
	}
	else
	{
		m_Angle = (int)(TmpAngle * 256.0f);
		//cout << m_Angle << endl;
	}

	return m_Angle + 402;
}

int calc_angles_distance(int angle_1, int angle_2)
{
	int max = 1205 + 402;
	int max_half = 804;

	int dist = abs(angle_1 - angle_2);

	if(dist >= max_half)
	{
		dist = max - dist;
	}

	return dist;
}

int CServer::TestNN()
{
	int score = 0;

	int start_tick = m_CurrentGameTick;

	while(true)
	{
		//GameServer()->OnPreTickTeehistorian();

		auto gamecontext = ((CGameContext *)GameServer());

		// Handle bots
		{
			auto gamelayer = gamecontext->Layers()->GameLayer();
			const CTile *pTiles = static_cast<CTile *>(Kernel()->RequestInterface<IMap>()->GetData(gamelayer->m_Data));

			// apply new input
			for(int c = 0; c < MAX_CLIENTS; c++)
			{
				if(m_aClients[c].m_State != CClient::STATE_INGAME)
					continue;

				// Move bot wherever you want
				if(strcmp(m_aClients[c].m_aName, "Bot") == 0)
				{
					vec2 center_coords = vec2(24.5f * 32.0f, 20.5f * 32.0f);

					auto bot_character = gamecontext->GetPlayerChar(c);

					if(bot_character != nullptr)
					{
						vec2 bot_pos = bot_character->m_Pos;
						/*std::cout << bot_pos.x
						     << " " << bot_pos.y << std::endl;*/

						vec2 delta_coords = center_coords - bot_pos;

						// bot_character->Core()->m_Angle

						int should_angle = coords_to_angle(delta_coords.x, delta_coords.y);
						int actual_angle = bot_character->Core()->m_Angle + 402;

						int reward = -calc_angles_distance(actual_angle, should_angle);
						// std::cout << actual_angle << " " << reward << std::endl;

						/*std::cout << delta_coords.x
							  << " " << delta_coords.y
							<< " Actual: " << angle
							     << " Should be: " << should_angle
							  << " Reward: " << reward << std::endl;*/
					}
				}

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
			}
		}

		for(int c = 0; c < MAX_CLIENTS; c++)
		{
			if(m_aClients[c].m_State != CClient::STATE_INGAME)
				continue;
			bool ClientHadInput = false;
			for(auto &Input : m_aClients[c].m_aInputs)
			{
				if(Input.m_GameTick == Tick() + 1)
				{
					GameServer()->OnClientPredictedEarlyInput(c, Input.m_aData);
					ClientHadInput = true;
				}
			}
			if(!ClientHadInput)
				GameServer()->OnClientPredictedEarlyInput(c, nullptr);
		}

		set_new_tick();

		m_CurrentGameTick++;

		// apply new input
		for(int c = 0; c < MAX_CLIENTS; c++)
		{
			if(m_aClients[c].m_State != CClient::STATE_INGAME)
				continue;

			// Move bot wherever you want
			if(strcmp(m_aClients[c].m_aName, "Bot") == 0)
			{
				// m_Direction:
				// 1 - Right
				// 0 - Stay
				// -1 - Left

				static int angle = 0;

				auto coords = angle_to_coords(angle);

				angle += 1;
				angle %= 1608;

				vec2 center_coords = vec2(24.5f * 32.0f, 20.5f * 32.0f);

				auto bot_character = gamecontext->GetPlayerChar(c);

				if(bot_character != nullptr)
				{
					vec2 bot_pos = bot_character->m_Pos;
					/*std::cout << bot_pos.x
					     << " " << bot_pos.y << std::endl;*/

					vec2 delta_coords = center_coords - bot_pos;

					int should_angle = coords_to_angle(delta_coords.x, delta_coords.y);

					int reward = -calc_angles_distance(angle, should_angle);

					/*std::cout << delta_coords.x
						  << " " << delta_coords.y
						<< " Actual: " << angle
						     << " Should be: " << should_angle
						  << " Reward: " << reward << std::endl;*/
				}

				CNetObj_PlayerInput pApplyInput;
				mem_zero(&pApplyInput, sizeof(pApplyInput));
				pApplyInput.m_TargetX = coords.x;
				pApplyInput.m_TargetY = coords.y;

				// auto gamecontext = ((CGameContext *)GameServer());

				// gamecontext->m_apPlayers[0]->m_Score;

				GameServer()->OnClientPredictedInput(c, &pApplyInput);

				// #include <>
				// if(strcmp(m_aClients[0].m_aName, "heartless tee") == 0)
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
				// }
				// else
				//{
				//	CNetObj_PlayerInput pApplyInput;
				//	mem_zero(&pApplyInput, sizeof(pApplyInput));
				//	pApplyInput.m_TargetX = coords.x;
				//	pApplyInput.m_TargetY = coords.y;

				//	//auto gamecontext = ((CGameContext *)GameServer());

				//	// gamecontext->m_apPlayers[0]->m_Score;

				//	GameServer()->OnClientPredictedInput(c, &pApplyInput);
				//}

				continue;
			}

			if(strcmp(m_aClients[c].m_aName, "heartless tee") == 0)
			{
				char buf[256];

				auto gamecontext = ((CGameContext *)GameServer());

				auto player_char = gamecontext->GetPlayerChar(c);

				if(player_char != nullptr)
				{
					auto gamelayer = gamecontext->Layers()->GameLayer();

					const CTile *pTiles = static_cast<CTile *>(Kernel()->RequestInterface<IMap>()->GetData(gamelayer->m_Data));

					const int Index = (int)(player_char->m_Pos.y / 32 + 1) * gamelayer->m_Width + (int)(player_char->m_Pos.x / 32);
					const int GameIndex = pTiles[Index].m_Index;
					sprintf(buf, "x:%f y:%f %i\n", player_char->m_Pos.x, player_char->m_Pos.y, GameIndex);
					printf(buf);
				}

				// for(int y = 0; y < gamelayer->m_Height; y++)
				//{
				//	for(int x = 0; x < gamelayer->m_Width; x++)
				//	{
				//		const int Index = y * gamelayer->m_Width + x;

				//		// Game layer
				//		{
				//			char buf[256];
				//			const int GameIndex = pTiles[Index].m_Index;

				//			sprintf(buf, "%i", GameIndex);

				//			printf("Hello\n");
				//		}
				//	}
				//}
			}

			bool ClientHadInput = false;
			for(auto &Input : m_aClients[c].m_aInputs)
			{
				if(Input.m_GameTick == Tick())
				{
					CNetObj_PlayerInput *pApplyInput = (CNetObj_PlayerInput *)Input.m_aData;
					// pApplyInput->m_Direction = 1;

					GameServer()->OnClientPredictedInput(c, pApplyInput);
					ClientHadInput = true;
					break;
				}
			}
			if(!ClientHadInput)
				GameServer()->OnClientPredictedInput(c, nullptr);
		}

		GameServer()->OnTick();
		if(ErrorShutdown())
		{
			break;
		}

		/*if(SpeedUpTicks)
		{
			break;
		}*/
	}

	return 0;
}

int CServer::Run()
{
	if(m_RunServer == UNINITIALIZED)
		m_RunServer = RUNNING;

	m_AuthManager.Init();

	unsigned int Seed;
	secure_random_fill(&Seed, sizeof(Seed));
	srand(Seed);

	if(Config()->m_Debug)
	{
		g_UuidManager.DebugDump();
	}

	{
		int Size = GameServer()->PersistentClientDataSize();
		for(auto &Client : m_aClients)
		{
			Client.m_HasPersistentData = false;
			Client.m_pPersistentData = malloc(Size);
		}
	}

	// load map
	if(!LoadMap(Config()->m_SvMap))
	{
		dbg_msg("server", "failed to load map. mapname='%s'", Config()->m_SvMap);
		return -1;
	}

	if(Config()->m_SvSqliteFile[0] != '\0')
	{
		char aFullPath[IO_MAX_PATH_LENGTH];
		Storage()->GetCompletePath(IStorage::TYPE_SAVE_OR_ABSOLUTE, Config()->m_SvSqliteFile, aFullPath, sizeof(aFullPath));

		if(Config()->m_SvUseSQL)
		{
			DbPool()->RegisterSqliteDatabase(CDbConnectionPool::WRITE_BACKUP, aFullPath);
		}
		else
		{
			DbPool()->RegisterSqliteDatabase(CDbConnectionPool::READ, aFullPath);
			DbPool()->RegisterSqliteDatabase(CDbConnectionPool::WRITE, aFullPath);
		}
	}

	// start server
	NETADDR BindAddr;
	int NetType = Config()->m_SvIpv4Only ? NETTYPE_IPV4 : NETTYPE_ALL;

	if(!Config()->m_Bindaddr[0] || net_host_lookup(Config()->m_Bindaddr, &BindAddr, NetType) != 0)
		mem_zero(&BindAddr, sizeof(BindAddr));

	BindAddr.type = NetType;

	int Port = Config()->m_SvPort;
	for(BindAddr.port = Port != 0 ? Port : 8303; !m_NetServer.Open(BindAddr, &m_ServerBan, Config()->m_SvMaxClients, Config()->m_SvMaxClientsPerIP); BindAddr.port++)
	{
		if(Port != 0 || BindAddr.port >= 8310)
		{
			dbg_msg("server", "couldn't open socket. port %d might already be in use", BindAddr.port);
			return -1;
		}
	}

	if(Port == 0)
		dbg_msg("server", "using port %d", BindAddr.port);

#if defined(CONF_UPNP)
	m_UPnP.Open(BindAddr);
#endif

	IEngine *pEngine = Kernel()->RequestInterface<IEngine>();
	m_pRegister = CreateRegister(&g_Config, m_pConsole, pEngine, this->Port(), m_NetServer.GetGlobalToken());

	m_NetServer.SetCallbacks(NewClientCallback, NewClientNoAuthCallback, ClientRejoinCallback, DelClientCallback, this);

	m_Econ.Init(Config(), Console(), &m_ServerBan);

#if defined(CONF_FAMILY_UNIX)
	m_Fifo.Init(Console(), Config()->m_SvInputFifo, CFGFLAG_SERVER);
#endif

	char aBuf[256];
	str_format(aBuf, sizeof(aBuf), "server name is '%s'", Config()->m_SvName);
	Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", aBuf);

	Antibot()->Init();
	GameServer()->OnInit();
	if(ErrorShutdown())
	{
		m_RunServer = STOPPING;
	}
	Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", "version " GAME_RELEASE_VERSION " on " CONF_PLATFORM_STRING " " CONF_ARCH_STRING);
	if(GIT_SHORTREV_HASH)
	{
		str_format(aBuf, sizeof(aBuf), "git revision hash: %s", GIT_SHORTREV_HASH);
		Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", aBuf);
	}

	// process pending commands
	m_pConsole->StoreCommands(false);
	m_pRegister->OnConfigChange();

	if(m_AuthManager.IsGenerated())
	{
		dbg_msg("server", "+-------------------------+");
		dbg_msg("server", "| rcon password: '%s' |", Config()->m_SvRconPassword);
		dbg_msg("server", "+-------------------------+");
	}

	int skip_tick = 4;
	int update_tick = 1000 * skip_tick;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> spawn_probabilities_distribution;

	std::vector<vec2> vSpawnPoints;
	std::vector <std::pair<int, int>> vFinishPoses;
	std::vector<std::vector<int>> pathfinding_grid;

	auto gamecontext = ((CGameContext *)GameServer());
	const CMapItemLayerTilemap *pTileMap = gamecontext->Layers()->GameLayer();
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
	printf("Creating pathfinder...\n");
	AStar astar(pathfinding_grid, vFinishPoses);
	printf("Pathfinder created.\n");

	printf("Creating train directory with folders...\n");

	static auto dir_name = to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());

	if(fs_makedir("train") != 0)
	{
		cout << "Can't make train directory" << endl;
		exit(1);
	}

	if(fs_makedir((string("train\\") + dir_name).c_str()) != 0)
	{
		cout << "Can't make dir for this learning directory" << endl;
		exit(1);
	}

	if(fs_makedir(string("train\\" + dir_name + "\\models").c_str()) != 0)
	{
		cout << "Can't make models directory" << endl;
		exit(1);
	}

	if(fs_makedir(string("train\\" + dir_name + "\\demos").c_str()) != 0)
	{
		cout << "Can't make demos directory" << endl;
		exit(1);
	}

	printf("Train directory with folders created.\n");

	//std::pair<int, int> start = {4, 4};
	//std::pair<int, int> goal = {4, 62};

	//std::vector<std::pair<int, int>> path = a_star(start, goal, pathfinding_grid);

	//if(!path.empty())
	//{
	//	std::cout << "Path found:\n";
	//	for(const auto &step : path)
	//	{
	//		std::cout << "(" << step.first << ", " << step.second << ")\n";
	//		pathfinding_grid[step.first][step.second] = 2;
	//	}
	//}
	//else
	//{
	//	std::cout << "No path found.\n";
	//}

	//// Display grid for pathfinding algorithm
	//for(size_t y = 0; y < pathfinding_grid.size(); y++)
	//{
	//	for(size_t x = 0; x < pathfinding_grid[y].size() / 2; x++)
	//	{
	//		cout << pathfinding_grid[y][x];
	//	}
	//	cout << endl;
	//}

	std::vector<float> vSpawnCumulativeReward(vSpawnPoints.size());
	std::vector<int> vSpawnLives(vSpawnPoints.size());
	std::vector<float> vSpawnProbabilities(vSpawnPoints.size(), 1);

	spawn_probabilities_distribution = std::discrete_distribution<>(vSpawnProbabilities.begin(), vSpawnProbabilities.end());
	printf("Adding bots...\n");
	//auto bot = AddBot("Bot");
	//auto bot_2 = AddBot("Bot2");
	int count_bots = 1024;
	std::vector<CPlayer *> vBots;
	std::vector<std::vector<std::pair<int, int>>> vBotsPath;
	std::vector<vec2> vBotLastPos;
	std::vector<int> vBotsSpawnPos;
	std::vector<vec2> vBotsLastCheckPoint;
	std::vector<float> vBotsCumulativeRewards;
	std::vector<ModelInputInputs> vInputInputs;
	std::vector<ModelInputBlocks> vInputBlocks;
	std::vector<ModelOutput> vOutputs;
	std::vector<bool> vIsPreviouslyHooked;
	std::vector<vec2> vPrevHookPos;

	// First is distance, second is tick
	std::vector<std::pair<int, int>> vBotBestDistance;

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
			vBotsPath[i] = astar.findPath(spawn_point_pos, 30);
			//auto now = time_get_impl();
			//cout << "Time to find: " << (float)(now - decide_time) / (float)time_freq() << " Length: " << vBotsPath[i].size() << endl;
			bot->KillCharacter();
			bot->TryRespawn(vSpawnPoints[iSpawnPoint]);
			bot->GetCharacter()->SetSolo(true);
			vBots.push_back(bot);
			vBotsLastCheckPoint[i] = bot->GetCharacter()->m_Pos;
			vBotsSpawnPos[i] = iSpawnPoint;
			vSpawnLives[iSpawnPoint] += 1;
			vBotBestDistance[i] = {astar.distanceToGoal(spawn_point_pos), 0};
			// auto tr = std::thread(RunNNForward, &model_manager, i, &vEvents, &vFinishEvents, &vInputs, &vOutputs);
			// tr.detach();
			//char aFilename[IO_MAX_PATH_LENGTH];
			//str_format(aFilename, sizeof(aFilename), "%s_%s_%d_%llu.demo", m_aCurrentMap, name.c_str(), m_NetServer.Address().port, time_get());
			//string path_demo = "train/" + dir_name + "/demos/" + aFilename;
			//int ret = m_aDemoRecorder[i].Start(Storage(), m_pConsole, path_demo.c_str(), GameServer()->NetVersion(), m_aCurrentMap, &m_aCurrentMapSha256[MAP_TYPE_SIX], m_aCurrentMapCrc[MAP_TYPE_SIX], "server", m_aCurrentMapSize[MAP_TYPE_SIX], m_apCurrentMapData[MAP_TYPE_SIX]);
		}
	}
	printf("Bots added\n");

	printf("Initializing neural model...\n");
	ModelManager model_manager(count_bots * update_tick / skip_tick, count_bots);
	printf("Model initialized.\n");

	printf("Creating data.csv file for statistics...\n");
	std::ofstream logger;
	{
		char aFilename[IO_MAX_PATH_LENGTH];
		/*std::cout << model_manager.GetLearningRate() << std::endl;
		std::cout << model_manager.GetMiniBatchSize() << std::endl;
		std::cout << model_manager.GetCountPPOEpochs() << std::endl;
		std::cout << count_bots << std::endl;
		std::cout << update_tick << std::endl;*/
		sprintf_s(aFilename, sizeof(aFilename), "lr%.1embs%lldppoe%lldbots%drpb%d.csv", model_manager.GetLearningRate(), model_manager.GetMiniBatchSize(), model_manager.GetCountPPOEpochs(), count_bots, update_tick);
		logger.open("train\\" + dir_name + "\\" + aFilename);
		logger << "Step,Average reward,TPS,Dies,Average distance,Training loss,Actor loss,Critic loss,Learning rate,Time since start,Time to decide,Time to tick,Time rest,Time pre forward,Time forward,Time normal,Time to cpu,Time process last" << endl;
	}

	printf("data.csv file created and initialized.\n");
	string path_demo;
	{
		char aFilename[IO_MAX_PATH_LENGTH];
		str_format(aFilename, sizeof(aFilename), "%s_%d_%llu.demo", m_aCurrentMap, m_NetServer.Address().port, time_get());
		path_demo = "train/" + dir_name + "/demos/" + aFilename;
		int ret = m_aDemoRecorder[0].Start(Storage(), m_pConsole, path_demo.c_str(), GameServer()->NetVersion(), m_aCurrentMap, &m_aCurrentMapSha256[MAP_TYPE_SIX], m_aCurrentMapCrc[MAP_TYPE_SIX], "server", m_aCurrentMapSize[MAP_TYPE_SIX], m_apCurrentMapData[MAP_TYPE_SIX]);
	}
	
	// start game
	{
		bool NonActive = false;
		bool PacketWaiting = false;
		bool SpeedUpTicks = true;

		int ticks_per_second = 0;
		int start_ticks = m_CurrentGameTick;

		// Neural network
		/*static constexpr int NumInput = 2;
		static constexpr int NumOutput = 1;
		static constexpr bool Bias = true;
		static constexpr float ThresholdFitness = -300.f;
		static constexpr std::size_t PopulationSize = 1000;
		auto rndSeed = ::GenerateSeed();

		using ParamConfig = EvolutionNet::DefaultParamConfig;
		using EvolutionNetT = EvolutionNet::EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>;
		using Network = EvolutionNetT::NetworkT;
		using FitnessScore = EvolutionNet::FitnessScore;

		EvolutionNetT evolutionNet;

		evolutionNet.initialize(PopulationSize, rndSeed);*/
		
		auto ticks_timer = time_get();

		m_GameStartTime = time_get();

		UpdateServerInfo();
		while(m_RunServer < STOPPING)
		{
			if(NonActive && !SpeedUpTicks)
				PumpNetwork(PacketWaiting);

			set_new_tick();
			//printf("1\n");
			int64_t t = time_get();
			int NewTicks = 0;
			//printf("2\n");
			// load new map
			if(m_MapReload || m_CurrentGameTick >= 0x6FFFFFFF) // force reload to make sure the ticks stay within a valid range
			{
				// load map
				if(LoadMap(Config()->m_SvMap))
				{
					// new map loaded

					// ask the game to for the data it wants to persist past a map change
					for(int i = 0; i < MAX_CLIENTS; i++)
					{
						if(m_aClients[i].m_State == CClient::STATE_INGAME)
						{
							m_aClients[i].m_HasPersistentData = GameServer()->OnClientDataPersist(i, m_aClients[i].m_pPersistentData);
						}
					}

					GameServer()->OnShutdown();

					for(int ClientID = 0; ClientID < MAX_CLIENTS; ClientID++)
					{
						if(m_aClients[ClientID].m_State <= CClient::STATE_AUTH)
							continue;

						SendMap(ClientID);
						bool HasPersistentData = m_aClients[ClientID].m_HasPersistentData;
						m_aClients[ClientID].Reset();
						m_aClients[ClientID].m_HasPersistentData = HasPersistentData;
						m_aClients[ClientID].m_State = CClient::STATE_CONNECTING;
					}

					m_GameStartTime = time_get();
					m_CurrentGameTick = 0;
					m_ServerInfoFirstRequest = 0;
					Kernel()->ReregisterInterface(GameServer());
					GameServer()->OnInit();
					if(ErrorShutdown())
					{
						break;
					}
					UpdateServerInfo(true);
				}
				else
				{
					str_format(aBuf, sizeof(aBuf), "failed to load map. mapname='%s'", Config()->m_SvMap);
					Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", aBuf);
					str_copy(Config()->m_SvMap, m_aCurrentMap);
				}
			}
			//printf("3\n");
			// handle dnsbl
			if(Config()->m_SvDnsbl)
			{
				for(int ClientID = 0; ClientID < MAX_CLIENTS; ClientID++)
				{
					if(m_aClients[ClientID].m_State == CClient::STATE_EMPTY)
						continue;

					if(m_aClients[ClientID].m_DnsblState == CClient::DNSBL_STATE_NONE)
					{
						// initiate dnsbl lookup
						InitDnsbl(ClientID);
					}
					else if(m_aClients[ClientID].m_DnsblState == CClient::DNSBL_STATE_PENDING &&
						m_aClients[ClientID].m_pDnsblLookup->Status() == IJob::STATE_DONE)
					{
						if(m_aClients[ClientID].m_pDnsblLookup->m_Result != 0)
						{
							// entry not found -> whitelisted
							m_aClients[ClientID].m_DnsblState = CClient::DNSBL_STATE_WHITELISTED;
						}
						else
						{
							// entry found -> blacklisted
							m_aClients[ClientID].m_DnsblState = CClient::DNSBL_STATE_BLACKLISTED;
							
							// console output
							char aAddrStr[NETADDR_MAXSTRSIZE];
							net_addr_str(m_NetServer.ClientAddr(ClientID), aAddrStr, sizeof(aAddrStr), true);

							str_format(aBuf, sizeof(aBuf), "ClientID=%d addr=<{%s}> secure=%s blacklisted", ClientID, aAddrStr, m_NetServer.HasSecurityToken(ClientID) ? "yes" : "no");

							Console()->Print(IConsole::OUTPUT_LEVEL_ADDINFO, "dnsbl", aBuf);
						}
					}

					if(m_aClients[ClientID].m_DnsblState == CClient::DNSBL_STATE_BLACKLISTED &&
						Config()->m_SvDnsblBan)
						m_NetServer.NetBan()->BanAddr(m_NetServer.ClientAddr(ClientID), 60 * 10, "VPN detected, try connecting without. Contact admin if mistaken");
				}
			}

			static bool started = false;

			static int start_tick = m_CurrentGameTick;
			//printf("5\n");
			while(t > TickStartTime(m_CurrentGameTick + 1) || SpeedUpTicks)
			{
				GameServer()->OnPreTickTeehistorian();

				auto gamecontext = ((CGameContext *)GameServer());

				static std::chrono::high_resolution_clock::time_point decide_time = std::chrono::high_resolution_clock::now();
				static float cumulative_time_to_decide = 0;
				static float cumulative_time_to_tick = 0;
				static float cumulative_time_rest = 0;

				// Decide times
				static double cumulative_time_pre_forward = 0;
				static double cumulative_time_forward = 0;
				static double cumulative_time_normal = 0;
				static double cumulative_time_to_cpu = 0;
				static double cumulative_time_process_last = 0;

				// Handle bots
				if(m_CurrentGameTick % skip_tick == 0)
				{
					auto gamelayer = gamecontext->Layers()->GameLayer();
					const CTile *pTiles = static_cast<CTile *>(Kernel()->RequestInterface<IMap>()->GetData(gamelayer->m_Data));
					int map_width = gamelayer->m_Width;
					int map_height = gamelayer->m_Height;
					//printf("6\n");
					static int dies = 0;
					static int moved_distance = 0;
					static float cumulative_reward = 0;

					// Rewards
					static float checkpoint_reward = 100.f / 32.f;
					static float die_reward = -250.f / 32.f; // -500.f / 32.f
					static float finish_reward = 1000.f / 32.f;
					static float step_reward = -0.01f;

					static std::vector<float> rewards;
					static float best_average = -999999.f;
					//printf("7\n");
					//printf("HAHHA1\n");
					//size_t summerr = 0;
					//decide_time = time_get_impl();
					for(size_t i = 0; i < vBots.size(); i++)
					{
						auto bot = vBots[i];
						// Move bot wherever you want
						//vec2 center_coords = vec2(12.f * 32.f, 12.f * 32.f);
						//printf("8\n");
						auto bot_character = bot->GetCharacter();
						//printf("9\n");
						// auto bot_2_character = gamecontext->GetPlayerChar(bot_2->GetCID());
						bool died = false;
						bool finished = false;
						bool freezed = false;

						if(bot_character != nullptr)
						{
							vec2 bot_pos = bot_character->Core()->m_Pos;

							int bot_block_pos_x = (int)(bot_pos.x / 32);
							int bot_block_pos_y = (int)(bot_pos.y / 32);
							int bot_block_index = bot_block_pos_y * map_width + bot_block_pos_x;

							if(pTiles[bot_block_index].m_Index == 34)
							{
								finished = true;
							}

							if(bot_character->m_FreezeTime || bot_character->Core()->m_IsInFreeze || pTiles[bot_block_index].m_Index == 9)
							{
								freezed = true;
							}
						}

						if(bot_character == nullptr || freezed || finished)
						{
							if(bot_character == nullptr || freezed)
							{
								died = true;
								dies += 1;
								if(bot_character)
								{
									bot->KillCharacter();
								}
							}
							else if(finished)
							{
								//m_aDemoRecorder[bot->GetCID()].Stop();

								/*char aNewFilename[IO_MAX_PATH_LENGTH];
								str_format(aNewFilename, sizeof(aNewFilename), "average_dist_%.2f_rew_%.2f_%s_%llu.demo", avg_dist, avg_reward, m_aCurrentMap, time_get_impl());
								path_demo = "train/" + dir_name + "/demos/" + aNewFilename;
								Storage()->RenameFile(m_aDemoRecorder[bot->GetCID()].GetCurrentFilename(), path_demo.c_str(), IStorage::TYPE_ABSOLUTE);*/

								bot->KillCharacter();

								/*char aFilename[IO_MAX_PATH_LENGTH];
								str_format(aFilename, sizeof(aFilename), "%s_%s_%d_%llu.demo", m_aCurrentMap, name.c_str(), m_NetServer.Address().port, time_get());
								string path_demo = "train/" + dir_name + "/demos/" + aFilename;
								int ret = m_aDemoRecorder[i].Start(Storage(), m_pConsole, path_demo.c_str(), GameServer()->NetVersion(), m_aCurrentMap, &m_aCurrentMapSha256[MAP_TYPE_SIX], m_aCurrentMapCrc[MAP_TYPE_SIX], "server", m_aCurrentMapSize[MAP_TYPE_SIX], m_apCurrentMapData[MAP_TYPE_SIX]);*/
							}
							
							//decide_time = time_get_impl();
							//bot_character->Core()->m_IsInFreeze
							//int iSpawnPoint = (int)round(random_float() * (float)vSpawnPoints.size()) % vSpawnPoints.size();
							int iSpawnPoint = spawn_probabilities_distribution(gen);
							bot->TryRespawn(vSpawnPoints[iSpawnPoint]);
							//summerr += time_get_impl() - decide_time;
							bot_character = bot->GetCharacter();
							bot_character->SetSolo(true);
							if(bot_character == nullptr)
							{
								cout << "FFFFUUUUUCCCCKKK" << endl;
								exit(1);
							}

							vBotsSpawnPos[i] = iSpawnPoint;
							vSpawnLives[iSpawnPoint] += 1;
							vBotsLastCheckPoint[i] = bot_character->m_PrevPos = bot_character->m_Pos = bot_character->Core()->m_Pos = vSpawnPoints[iSpawnPoint];
							bot_character->Core()->m_Vel = vec2(2.f * random_float() - 1.f, 2.f * random_float() - 1.f);
							vBotsCumulativeRewards[i] = 0;
							auto spawn_point_pos = std::pair<int, int>((int)vSpawnPoints[iSpawnPoint].y / 32, (int)vSpawnPoints[iSpawnPoint].x / 32);
							vBotsPath[i] = astar.findPath(spawn_point_pos, 30);
							vBotBestDistance[i] = {astar.distanceToGoal(spawn_point_pos), m_CurrentGameTick};
							//bot_character->Core()->m_Pos.x = 3.f * 32.f + random_float() * 4.f * 32.f;
						}
						
						//printf("10\n");
						if(started)
						{
							float reward = 0.f;

							vec2 bot_pos = bot_character->Core()->m_Pos;

							int bot_block_pos_x = (int)(bot_pos.x / 32);
							int bot_block_pos_y = (int)(bot_pos.y / 32);
							int bot_block_index = bot_block_pos_y * map_width + bot_block_pos_x;

							if(died)
							{
								reward += die_reward; // 1000 500 348 * 32;
							}
							/*else if(bot_character->m_Pos.x - vBotsSpawnPos[i].x >= 50.f * 32.f)
							{
								vBotsSpawnPos[i] = bot_character->m_Pos;
								reward += 348 * 32;
							}*/
							else if(finished)
							{
								reward += finish_reward; // 348 * 32;
							}
							//else if(bot_character->IsGrounded())
							//{
							//	reward = -20;
							//}
							else
							{
								// If touched the checkpoint reward
								if(pTiles[bot_block_index].m_Index == 35 && vBotsLastCheckPoint[i].x < bot_block_pos_x * 32.f)
								{
									reward += checkpoint_reward; // 20 50 96 * 32;
									vBotsLastCheckPoint[i] = vec2(bot_block_pos_x * 32.f, bot_block_pos_y * 32.f);
								}
								//reward += bot_character->m_Pos.x - bot_character->m_PrevPos.x;
								int bot_last_block_pos_x = vBotLastPos[i].x / 32;
								int bot_last_block_pos_y = vBotLastPos[i].y / 32;
								if(bot_last_block_pos_x != bot_block_pos_x || bot_last_block_pos_y != bot_block_pos_y)
								{
									int prev_dist = astar.distanceToGoal(bot_last_block_pos_y, bot_last_block_pos_x);
									vBotsPath[i] = astar.findPath(std::pair<int, int>(bot_block_pos_y, bot_block_pos_x), 30);
									int current_dist = astar.distanceToGoal(bot_block_pos_y, bot_block_pos_x);
									int path_dist_diff = prev_dist - current_dist;
									reward += path_dist_diff;
									moved_distance += path_dist_diff;
									if(current_dist > vBotBestDistance[i].first)
									{
										vBotBestDistance[i] = {current_dist, m_CurrentGameTick};
									}

								}

								auto tick_diff = m_CurrentGameTick - vBotBestDistance[i].second;

								auto long_stay_penalty = -0.0003f * tick_diff;

								//cout << reward << endl;

								/*if(reward < 0)
								{
									reward *= 32;
								}

								if(reward == 0)
								{
									reward -= 32;
								}*/

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

							bool is_done = died || finished || (m_CurrentGameTick % 1000 == 0) ? 1 : 0;
							model_manager.Reward(reward, is_done);
							if(!died && !finished)
							{
								vBotsCumulativeRewards[i] += reward;
							}
							char reward_name[16];
							sprintf_s(reward_name, "%.1f", vBotsCumulativeRewards[i]);
							memcpy(m_aClients[bot->GetCID()].m_aName, reward_name, strlen(reward_name) + 1);
							//std::cout << reward << std::endl;
							rewards.push_back(reward);
						}
					}
					//cout << "Time rewards: " << (float)(time_get_impl() - decide_time) / (float)time_freq() << endl;
					//printf("11\n");
					model_manager.SaveReplays();
					//cout << m_CurrentGameTick << endl;

					if(m_CurrentGameTick != 0 && m_CurrentGameTick % update_tick == 0)
					{
						//decide_time = time_get_impl();
						//printf("UPDATING\n");
						float avg_reward = cumulative_reward / (float)(dies + count_bots);
						float avg_dist = ((float)moved_distance / (float)(dies + count_bots));
						rewards.clear();

						if(m_CurrentGameTick % (20000 * skip_tick) == 0 && model_manager.IsTraining())
						{
							printf("UPDATING\n");
							std::vector<float> vAverageDistancePerSpawn(vSpawnCumulativeReward.size());
							int count_counted = 0;
							float cumulative_reward = 0;
							//printf("1\n");
							for(size_t i = 0; i < vSpawnCumulativeReward.size(); i++)
							{
								if(vSpawnLives[i] && vSpawnCumulativeReward[i] != 0.f)
								{
									//printf("1.1\n");
									float dist = vSpawnCumulativeReward[i] / (float)vSpawnLives[i];
									//printf("1.2\n");
									//std::cout << "vSpawnCumulativeDistance: " << vSpawnCumulativeDistance[i] << endl;
									//std::cout << "vSpawnLives[i]: " << vSpawnLives[i] << endl;
									//std::cout << "Dist: " << dist << endl;
									cumulative_reward += dist;
									//printf("1.3\n");
									vAverageDistancePerSpawn[i] = dist;
									//printf("1.4\n");
									count_counted += 1;
								}
							}
							//printf("1.9\n");
							float average_reward = 0;
							if(cumulative_reward != 0.f && count_counted)
							{
								average_reward = cumulative_reward / (float)count_counted;
							}
							cout << "Average reward: " << average_reward << endl;
							//printf("1.99\n");
							float max_reward = *max_element(vAverageDistancePerSpawn.begin(), vAverageDistancePerSpawn.end());
							cout << "Max reward: " << max_reward << endl;
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

						auto demo_recorder = &m_aDemoRecorder[0];

						if(demo_recorder->IsRecording() && model_manager.IsTraining())
						{
							demo_recorder->Stop();
							char aNewFilename[IO_MAX_PATH_LENGTH];
							str_format(aNewFilename, sizeof(aNewFilename), "average_dist_%.2f_rew_%.2f_%s_%llu.demo", avg_dist, avg_reward, m_aCurrentMap, time_get_impl());
							path_demo = "train/" + dir_name + "/demos/" + aNewFilename;
							Storage()->RenameFile(demo_recorder->GetCurrentFilename(), path_demo.c_str(), IStorage::TYPE_ABSOLUTE);
						}
						//printf("111\n");
						if(m_CurrentGameTick % (20000 * skip_tick) == 0 && model_manager.IsTraining())
						{
							model_manager.Save("train\\" + dir_name + "\\models\\last");
						}
						//printf("222\n");
						if(avg_dist > best_average && model_manager.IsTraining())
						{
							best_average = avg_dist;
							model_manager.Save("train\\" + dir_name + "\\models\\best"); // best" + to_string(average)
							
							/*namespace fs = std::filesystem;
							fs::path demo_path = fs::current_path() / "train" / dir_name / "demos";
							std::cout << demo_path.c_str() << std::endl;*/
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
						//printf("ret: %i\n", ret);
						//printf("start_u\n");
						//int64_t update_time = time_get_impl();
						double avg_training_loss = 0;
						double avg_actor_loss = 0;
						double avg_critic_loss = 0;
						model_manager.Update(avg_dist, avg_training_loss, avg_actor_loss, avg_critic_loss);
						//cout << "Time update: " << (float)(time_get_impl() - decide_time) / (float)time_freq() << endl;
						logger << m_CurrentGameTick / update_tick
						       << "," << avg_reward
						       << "," << ticks_per_second
						       << "," << dies
						       << "," << avg_dist
						       << "," << avg_training_loss
						       << "," << avg_actor_loss
						       << "," << avg_critic_loss
						       << "," << model_manager.GetCurrentLearningRate()
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
						cout << "Avg. reward: " << avg_reward << " TPS: " << ticks_per_second << " Avg. Training Loss: " << avg_training_loss
						     << " Dies: " << dies << " Avg. distance: " << avg_dist << endl;
						dies \
							= moved_distance \
							= cumulative_reward \
							= cumulative_time_to_decide \
							= cumulative_time_to_tick \
							= cumulative_time_rest \
							= cumulative_time_pre_forward \
							= cumulative_time_forward \
							= cumulative_time_normal \
							= cumulative_time_normal \
							= cumulative_time_to_cpu \
							= cumulative_time_process_last \
							= 0;
						//cout << "Time to update: " << (float)(time_get_impl() - update_time) / (float)time_freq() << endl;
						//printf("end\n");
						if(m_CurrentGameTick % 20000 == 0 && model_manager.IsTraining())
						{
							char aFilename[IO_MAX_PATH_LENGTH];
							str_format(aFilename, sizeof(aFilename), "%s_%d_%llu.demo", m_aCurrentMap, m_NetServer.Address().port, time_get());
							path_demo = "train\\" + dir_name + "\\demos\\" + aFilename;
							int ret = demo_recorder->Start(Storage(), m_pConsole, path_demo.c_str(), GameServer()->NetVersion(), m_aCurrentMap, &m_aCurrentMapSha256[MAP_TYPE_SIX], m_aCurrentMapCrc[MAP_TYPE_SIX], "server", m_aCurrentMapSize[MAP_TYPE_SIX], m_apCurrentMapData[MAP_TYPE_SIX]);
						}
						decide_time = std::chrono::high_resolution_clock::now();
					}
					//printf("12\n");
					// apply new input
					//decide_time = time_get_impl();
					for(size_t i = 0; i < vBots.size(); i++)
					{
						//printf("HAHHA3.1\n");
						auto bot = vBots[i];
						//printf("HAHHA3.2\n");
						// Move bot wherever you want
						//vec2 center_coords = vec2(12.f * 32.f, 12.f * 32.f);

						auto bot_character = bot->GetCharacter();
						auto bot_character_core = bot_character->Core();
						// auto bot_2_character = gamecontext->GetPlayerChar(bot_2->GetCID());
						//printf("HAHHA3.3\n");
						if(m_CurrentGameTick != 0 && m_CurrentGameTick % update_tick == 0 && model_manager.IsTraining())
						{
							// Add to cumulative spawn distance vector
							//int iOldSpawnPoint = vBotsSpawnPos[i];
							//vSpawnCumulativeDistance[iOldSpawnPoint] += bot_character_core->m_Pos.x - vSpawnPoints[iOldSpawnPoint].x;

							//int iSpawnPoint = (int)round(random_float() * (float)vSpawnPoints.size()) % vSpawnPoints.size();
							int iSpawnPoint = spawn_probabilities_distribution(gen);
							bot->KillCharacter();
							bot->TryRespawn(vSpawnPoints[iSpawnPoint]);

							bot_character = bot->GetCharacter();
							bot_character_core = bot_character->Core();

							bot_character->SetSolo(true);
							
							vBotsSpawnPos[i] = iSpawnPoint;
							vSpawnLives[iSpawnPoint] += 1;
							vBotsLastCheckPoint[i] = bot_character->m_PrevPos = bot_character->m_Pos = bot_character_core->m_Pos = vSpawnPoints[iSpawnPoint];
							bot_character_core->m_Vel = vec2(2.f * random_float() - 1.f, 2.f * random_float() - 1.f);
							vBotsCumulativeRewards[i] = 0;
							auto spawn_point_pos = std::pair<int, int>((int)vSpawnPoints[iSpawnPoint].y / 32, (int)vSpawnPoints[iSpawnPoint].x / 32);
							vBotBestDistance[i] = {astar.distanceToGoal(spawn_point_pos), m_CurrentGameTick};
							vBotsPath[i] = astar.findPath(spawn_point_pos, 30);
							// bot_2->KillCharacter();
							// bot_2->TryRespawn();
							//bot_character->Core()->m_Pos.x = 3.f * 32.f + random_float() * 4.f * 32.f;
							// bot_2_character->Core()->m_Pos.x = 4.f * 32.f + random_float() * 12.f * 32.f;
							start_tick = m_CurrentGameTick;
						}

						vec2 bot_pos = bot_character->Core()->m_Pos;
						/*std::cout << bot_pos.x
								<< " " << bot_pos.y << std::endl;*/
						// bot_character->Core()->m_HookDir
						//vec2 delta_coords = center_coords - bot_pos;

						/*vec2 rand_pos = {random_float() * 24.f * 32.f, random_float() * 24.f * 32.f};
						delta_coords = center_coords - rand_pos;

						int should_angle = coords_to_angle(delta_coords.x, delta_coords.y);
						int actual_angle = bot_character_core->m_Angle+402;
						prev_angle_dist = calc_angles_distance(actual_angle, should_angle);*/

						ModelInputInputs *input_inputs = &vInputInputs[i];
						ModelInputBlocks *input_blocks = &vInputBlocks[i];

						//auto gamecontext = ((CGameContext *)GameServer());

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
						input_inputs->is_retracted = bot_character_core->m_HookState == HOOK_RETRACTED
												 || (bot_character_core->m_HookState >= HOOK_RETRACT_START && bot_character_core->m_HookState <= HOOK_RETRACT_END);
						
						if(input_inputs->is_hooking)
						{
							//printf("HJQWHEe1\n");
							auto hook_relative = (bot_character_core->m_HookPos - bot_character_core->m_Pos) / gamecontext->Tuning()->m_HookLength;
							//printf("HJQWHEe2\n");
							input_inputs->hook_pos = vec2(std::clamp(hook_relative.x, -1.f, 1.f), std::clamp(hook_relative.y, -1.f, 1.f));
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
						//decide_time = time_get_impl();

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
							//cout << "x: " << x_move << " y: " << y_move << endl;
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
							//cout << endl;
						}
						//summerr += time_get_impl() - decide_time;


						//printf("HAHHA3.4\n");
						//vInputs[i] = input;
						//printf("HAHHA3.4.1\n");
						//printf("HAHHA3.5\n");
						// int64_t decide_time = time_get_impl();
						//ModelOutput returned_model = model_manager.Decide(input); // {0.0f}; //
						//// cout << "Time to decide: " << (float)(time_get_impl() - decide_time) / (float)time_freq() << endl;
						//model_angle = returned_model.angle * 1608;
						//model_direction = returned_model.direction;
						//model_hook = returned_model.hook;
						//started = true;
						// prev_angle_dist = calc_angles_distance((int)model_angle, should_angle);

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
					//cout << "Time calcs: " << (float)summerr / (float)time_freq() << endl;

					//auto player_char = gamecontext->GetPlayerChar(c);

					//if(player_char != nullptr)
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
					started = true;
					auto now = std::chrono::high_resolution_clock::now();
					cumulative_time_rest += std::chrono::duration_cast<std::chrono::duration<float>>(now - decide_time).count() * 1000.f;
					// cout << "Time rest: " << std::chrono::duration_cast<std::chrono::duration<float>>(now - decide_time).count() << endl;
					double time_pre_forward = 0;
					double time_forward = 0;
					double time_normal = 0;
					double time_to_cpu = 0;
					double time_process_last = 0;
					decide_time = std::chrono::high_resolution_clock::now();
					vOutputs = model_manager.Decide(vInputInputs, vInputBlocks, time_pre_forward, time_forward, time_normal, time_to_cpu, time_process_last);
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
				//printf("13\n");
				
				//printf("KEK\n");
				//printf("14\n");
				for(int c = 0; c < MAX_CLIENTS; c++)
				{
					if(m_aClients[c].m_State != CClient::STATE_INGAME)
						continue;
					bool ClientHadInput = false;
					for(auto &Input : m_aClients[c].m_aInputs)
					{
						if(Input.m_GameTick == Tick() + 1)
						{
							GameServer()->OnClientPredictedEarlyInput(c, Input.m_aData);
							ClientHadInput = true;
						}
					}
					if(!ClientHadInput)
						GameServer()->OnClientPredictedEarlyInput(c, nullptr);
				}
				
				//printf("15\n");
				if((time_get() - ticks_timer) / time_freq() >= 1.0f)
				{
					ticks_per_second = m_CurrentGameTick - start_ticks;
					//std::cout << "TPS: " << ticks_per_second << std::endl;
					start_ticks = m_CurrentGameTick;
					ticks_timer = time_get();
				}
				//printf("HAHHA_2\n");
				m_CurrentGameTick++;
				NewTicks++;
				//printf("16\n");
				for(size_t i = 0; i < vBots.size(); i++)
				{
					// printf("HAHHA1111\n");
					auto bot = vBots[i];
					// printf("HAHHA2222\n");
					//  Move bot wherever you want
					
					// m_Direction:
					// 1 - Right
					// 0 - Stay
					// -1 - Left

					int angle = 0;

					auto coords = angle_to_coords(angle);

					// angle += 1;
					// angle %= 1608;

					vec2 center_coords = vec2(24.5f * 32.0f, 20.5f * 32.0f);

					auto bot_character = bot->GetCharacter();
					int model_jump = 0;
					// printf("HAHHA2222\n");
					vec2 model_angle;
					int model_direction = 0;
					int model_hook = 0;

					if(bot_character != nullptr)
					{
						vec2 bot_pos = bot_character->m_Pos;
						/*std::cout << bot_pos.x
							    << " " << bot_pos.y << std::endl;*/

						vec2 delta_coords = center_coords - bot_pos;

						/*if(bot_character->IsGrounded())
						{
							jump = 1;
						}*/
						ModelOutput returned_model = vOutputs[i]; // {0.0f}; //
						// printf("HAHHA33.2\n");
						//  cout << "Time to decide: " << (float)(time_get_impl() - decide_time) / (float)time_freq() << endl;
						model_angle = returned_model.angle * 299.f;
						model_direction = returned_model.direction;
						model_hook = returned_model.hook;
						
						if(m_CurrentGameTick % skip_tick != 1)
						{
							model_jump = 0;
						}
						else
						{
							model_jump = returned_model.jump;
						}
						// printf("HAHHA33.3\n");
						// printf("HAHHA34\n");
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

						int should_angle = coords_to_angle(delta_coords.x, delta_coords.y);

						// ModelOutput returned_model = model_manager.Decide({bot_pos}); // {0.0f}; //
						// int model_angle = returned_model.angle * 1608;

						//int reward = -calc_angles_distance(model_angle, should_angle);

						// model_manager.Reward(reward, (m_CurrentGameTick - start_tick >= 1000) ? 1 : 0);
						// rewards.push_back(reward);

						//coords = angle_to_coords(model_angle);

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
					GameServer()->OnClientPredictedInput(bot->GetCID(), &pApplyInput);
					//printf("18\n");
					// #include <>
					// if(strcmp(m_aClients[0].m_aName, "heartless tee") == 0)
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
					// }
					// else
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
				//printf("19\n");
				// apply new input
				for(int c = 0; c < MAX_CLIENTS; c++)
				{
					if(m_aClients[c].m_State != CClient::STATE_INGAME)
						continue;

					if(strcmp(m_aClients[c].m_aName, "nameless tee") == 0)
					{
						auto gamecontext = ((CGameContext *)GameServer());

						auto player_char = gamecontext->GetPlayerChar(c);

						if(player_char != nullptr)
						{
							auto gamelayer = gamecontext->Layers()->GameLayer();

							const CTile *pTiles = static_cast<CTile *>(Kernel()->RequestInterface<IMap>()->GetData(gamelayer->m_Data));

							const int Index = (int)(player_char->m_Pos.y / 32 + 1) * gamelayer->m_Width + (int)(player_char->m_Pos.x / 32);
							const int GameIndex = pTiles[Index].m_Index;
							//printf("x:%f y:%f %i\n", player_char->m_Pos.x, player_char->m_Pos.y, GameIndex);
							//printf("x:%f y:%f\n", player_char->Core()->m_HookPos.x, player_char->Core()->m_HookPos.y);
							//player_char->IsGrounded();

							//printf("Jumps: %i\n", player_char->GetCore().m_Jumps);
							//printf("Grounded: %f\n", player_char->GetCore().m_Vel.x);
						}

						//for(int y = 0; y < gamelayer->m_Height; y++)
						//{
						//	for(int x = 0; x < gamelayer->m_Width; x++)
						//	{
						//		const int Index = y * gamelayer->m_Width + x;

						//		// Game layer
						//		{
						//			char buf[256];
						//			const int GameIndex = pTiles[Index].m_Index;

						//			sprintf(buf, "%i", GameIndex);

						//			printf("Hello\n");
						//		}
						//	}
						//}
					}

					bool ClientHadInput = false;
					for(auto &Input : m_aClients[c].m_aInputs)
					{
						if(Input.m_GameTick == Tick())
						{
							CNetObj_PlayerInput *pApplyInput = (CNetObj_PlayerInput *)Input.m_aData;
							//pApplyInput->m_Direction = 1;

							GameServer()->OnClientPredictedInput(c, pApplyInput);
							ClientHadInput = true;
							break;
						}
					}
					if(!ClientHadInput)
						GameServer()->OnClientPredictedInput(c, nullptr);
				}
				//printf("HAHHA_3\n");
				//printf("20\n");
				GameServer()->OnTick();
				if(ErrorShutdown())
				{
					break;
				}
				//printf("21\n");
				auto now = std::chrono::high_resolution_clock::now();
				cumulative_time_to_tick += std::chrono::duration_cast<std::chrono::duration<float>>(now - decide_time).count() * 1000.f;
				//cout << "Time to tick: " << (float)(now - decide_time) / (float)time_freq() << endl;
				decide_time = std::chrono::high_resolution_clock::now();

				/*if(m_CurrentGameTick == 1000)
				{
					m_aDemoRecorder[MAX_CLIENTS-1].Stop();
					exit(0);
				}*/

				if(SpeedUpTicks)
				{
					break;
				}
			}

			// snap game
			if(NewTicks)
			{
				if((Config()->m_SvHighBandwidth || (m_CurrentGameTick % 2) == 0) && m_aDemoRecorder[0].IsRecording())
					DoSnapshot();

				UpdateClientRconCommands();

#if defined(CONF_FAMILY_UNIX)
				m_Fifo.Update();
#endif
			}

			// master server stuff
			m_pRegister->Update();

			if(m_ServerInfoNeedsUpdate)
				UpdateServerInfo();

			//Antibot()->OnEngineTick();

			if(!NonActive && !SpeedUpTicks)
				PumpNetwork(PacketWaiting);

			NonActive = true;

			for(const auto &Client : m_aClients)
			{
				if(Client.m_State != CClient::STATE_EMPTY)
				{
					NonActive = false;
					break;
				}
			}

			if(!SpeedUpTicks)
			{
				// wait for incoming data
				if(NonActive)
				{
					if(Config()->m_SvReloadWhenEmpty == 1)
					{
						m_MapReload = true;
						Config()->m_SvReloadWhenEmpty = 0;
					}
					else if(Config()->m_SvReloadWhenEmpty == 2 && !m_ReloadedWhenEmpty)
					{
						m_MapReload = true;
						m_ReloadedWhenEmpty = true;
					}

					if(Config()->m_SvShutdownWhenEmpty)
						m_RunServer = STOPPING;
					else
						PacketWaiting = net_socket_read_wait(m_NetServer.Socket(), 1000000);
				}
				else
				{
					m_ReloadedWhenEmpty = false;

					set_new_tick();
					t = time_get();
					int x = (TickStartTime(m_CurrentGameTick + 1) - t) * 1000000 / time_freq() + 1;

					PacketWaiting = x > 0 ? net_socket_read_wait(m_NetServer.Socket(), x) : true;
				}
			}

			if(IsInterrupted())
			{
				Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", "interrupted");
				break;
			}
		}
	}
	const char *pDisconnectReason = "Server shutdown";
	if(m_aShutdownReason[0])
		pDisconnectReason = m_aShutdownReason;

	if(ErrorShutdown())
	{
		dbg_msg("server", "shutdown from game server (%s)", m_aErrorShutdownReason);
		pDisconnectReason = m_aErrorShutdownReason;
	}
	// disconnect all clients on shutdown
	for(int i = 0; i < MAX_CLIENTS; ++i)
	{
		if(m_aClients[i].m_State != CClient::STATE_EMPTY)
			m_NetServer.Drop(i, pDisconnectReason);
	}

	m_Econ.Shutdown();

#if defined(CONF_FAMILY_UNIX)
	m_Fifo.Shutdown();
#endif

	GameServer()->OnShutdown();
	m_pMap->Unload();

	DbPool()->OnShutdown();

#if defined(CONF_UPNP)
	m_UPnP.Shutdown();
#endif

	m_NetServer.Close();

	m_pRegister->OnShutdown();

	return ErrorShutdown();
}

void CServer::ConTestingCommands(CConsole::IResult *pResult, void *pUser)
{
	CConsole *pThis = static_cast<CConsole *>(pUser);
	char aBuf[128];
	str_format(aBuf, sizeof(aBuf), "Value: %d", pThis->Config()->m_SvTestingCommands);
	pThis->Print(CConsole::OUTPUT_LEVEL_STANDARD, "console", aBuf);
}

void CServer::ConRescue(CConsole::IResult *pResult, void *pUser)
{
	CConsole *pThis = static_cast<CConsole *>(pUser);
	char aBuf[128];
	str_format(aBuf, sizeof(aBuf), "Value: %d", pThis->Config()->m_SvRescue);
	pThis->Print(CConsole::OUTPUT_LEVEL_STANDARD, "console", aBuf);
}

void CServer::ConKick(IConsole::IResult *pResult, void *pUser)
{
	if(pResult->NumArguments() > 1)
	{
		char aBuf[128];
		str_format(aBuf, sizeof(aBuf), "Kicked (%s)", pResult->GetString(1));
		((CServer *)pUser)->Kick(pResult->GetInteger(0), aBuf);
	}
	else
		((CServer *)pUser)->Kick(pResult->GetInteger(0), "Kicked by console");
}

void CServer::ConStatus(IConsole::IResult *pResult, void *pUser)
{
	char aBuf[1024];
	char aAddrStr[NETADDR_MAXSTRSIZE];
	CServer *pThis = static_cast<CServer *>(pUser);
	const char *pName = pResult->NumArguments() == 1 ? pResult->GetString(0) : "";

	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(pThis->m_aClients[i].m_State == CClient::STATE_EMPTY)
			continue;

		if(!str_utf8_find_nocase(pThis->m_aClients[i].m_aName, pName))
			continue;

		net_addr_str(pThis->m_NetServer.ClientAddr(i), aAddrStr, sizeof(aAddrStr), true);
		if(pThis->m_aClients[i].m_State == CClient::STATE_INGAME)
		{
			char aDnsblStr[64];
			aDnsblStr[0] = '\0';
			if(pThis->Config()->m_SvDnsbl)
			{
				const char *pDnsblStr = pThis->m_aClients[i].m_DnsblState == CClient::DNSBL_STATE_WHITELISTED ? "white" :
																pThis->m_aClients[i].m_DnsblState == CClient::DNSBL_STATE_BLACKLISTED ? "black" :
																									pThis->m_aClients[i].m_DnsblState == CClient::DNSBL_STATE_PENDING ? "pending" : "n/a";

				str_format(aDnsblStr, sizeof(aDnsblStr), " dnsbl=%s", pDnsblStr);
			}

			char aAuthStr[128];
			aAuthStr[0] = '\0';
			if(pThis->m_aClients[i].m_AuthKey >= 0)
			{
				const char *pAuthStr = pThis->m_aClients[i].m_Authed == AUTHED_ADMIN ? "(Admin)" :
												       pThis->m_aClients[i].m_Authed == AUTHED_MOD ? "(Mod)" :
																		     pThis->m_aClients[i].m_Authed == AUTHED_HELPER ? "(Helper)" : "";

				str_format(aAuthStr, sizeof(aAuthStr), " key=%s %s", pThis->m_AuthManager.KeyIdent(pThis->m_aClients[i].m_AuthKey), pAuthStr);
			}

			const char *pClientPrefix = "";
			if(pThis->m_aClients[i].m_Sixup)
			{
				pClientPrefix = "0.7:";
			}
			str_format(aBuf, sizeof(aBuf), "id=%d addr=<{%s}> name='%s' client=%s%d secure=%s flags=%d%s%s",
				i, aAddrStr, pThis->m_aClients[i].m_aName, pClientPrefix, pThis->m_aClients[i].m_DDNetVersion,
				pThis->m_NetServer.HasSecurityToken(i) ? "yes" : "no", pThis->m_aClients[i].m_Flags, aDnsblStr, aAuthStr);
		}
		else
		{
			str_format(aBuf, sizeof(aBuf), "id=%d addr=<{%s}> connecting", i, aAddrStr);
		}
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", aBuf);
	}
}

static int GetAuthLevel(const char *pLevel)
{
	int Level = -1;
	if(!str_comp_nocase(pLevel, "admin"))
		Level = AUTHED_ADMIN;
	else if(str_startswith(pLevel, "mod"))
		Level = AUTHED_MOD;
	else if(!str_comp_nocase(pLevel, "helper"))
		Level = AUTHED_HELPER;

	return Level;
}

void CServer::AuthRemoveKey(int KeySlot)
{
	int NewKeySlot = KeySlot;
	int OldKeySlot = m_AuthManager.RemoveKey(KeySlot);
	LogoutKey(KeySlot, "key removal");

	// Update indices.
	if(OldKeySlot != NewKeySlot)
	{
		for(auto &Client : m_aClients)
			if(Client.m_AuthKey == OldKeySlot)
				Client.m_AuthKey = NewKeySlot;
	}
}

void CServer::ConAuthAdd(IConsole::IResult *pResult, void *pUser)
{
	CServer *pThis = (CServer *)pUser;
	CAuthManager *pManager = &pThis->m_AuthManager;

	const char *pIdent = pResult->GetString(0);
	const char *pLevel = pResult->GetString(1);
	const char *pPw = pResult->GetString(2);

	int Level = GetAuthLevel(pLevel);
	if(Level == -1)
	{
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "level can be one of {\"admin\", \"mod(erator)\", \"helper\"}");
		return;
	}

	bool NeedUpdate = !pManager->NumNonDefaultKeys();
	if(pManager->AddKey(pIdent, pPw, Level) < 0)
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "ident already exists");
	else
	{
		if(NeedUpdate)
			pThis->SendRconType(-1, true);
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "key added");
	}
}

void CServer::ConAuthAddHashed(IConsole::IResult *pResult, void *pUser)
{
	CServer *pThis = (CServer *)pUser;
	CAuthManager *pManager = &pThis->m_AuthManager;

	const char *pIdent = pResult->GetString(0);
	const char *pLevel = pResult->GetString(1);
	const char *pPw = pResult->GetString(2);
	const char *pSalt = pResult->GetString(3);

	int Level = GetAuthLevel(pLevel);
	if(Level == -1)
	{
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "level can be one of {\"admin\", \"mod(erator)\", \"helper\"}");
		return;
	}

	MD5_DIGEST Hash;
	unsigned char aSalt[SALT_BYTES];

	if(md5_from_str(&Hash, pPw))
	{
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "Malformed password hash");
		return;
	}
	if(str_hex_decode(aSalt, sizeof(aSalt), pSalt))
	{
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "Malformed salt hash");
		return;
	}

	bool NeedUpdate = !pManager->NumNonDefaultKeys();

	if(pManager->AddKeyHash(pIdent, Hash, aSalt, Level) < 0)
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "ident already exists");
	else
	{
		if(NeedUpdate)
			pThis->SendRconType(-1, true);
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "key added");
	}
}

void CServer::ConAuthUpdate(IConsole::IResult *pResult, void *pUser)
{
	CServer *pThis = (CServer *)pUser;
	CAuthManager *pManager = &pThis->m_AuthManager;

	const char *pIdent = pResult->GetString(0);
	const char *pLevel = pResult->GetString(1);
	const char *pPw = pResult->GetString(2);

	int KeySlot = pManager->FindKey(pIdent);
	if(KeySlot == -1)
	{
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "ident couldn't be found");
		return;
	}

	int Level = GetAuthLevel(pLevel);
	if(Level == -1)
	{
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "level can be one of {\"admin\", \"mod(erator)\", \"helper\"}");
		return;
	}

	pManager->UpdateKey(KeySlot, pPw, Level);
	pThis->LogoutKey(KeySlot, "key update");

	pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "key updated");
}

void CServer::ConAuthUpdateHashed(IConsole::IResult *pResult, void *pUser)
{
	CServer *pThis = (CServer *)pUser;
	CAuthManager *pManager = &pThis->m_AuthManager;

	const char *pIdent = pResult->GetString(0);
	const char *pLevel = pResult->GetString(1);
	const char *pPw = pResult->GetString(2);
	const char *pSalt = pResult->GetString(3);

	int KeySlot = pManager->FindKey(pIdent);
	if(KeySlot == -1)
	{
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "ident couldn't be found");
		return;
	}

	int Level = GetAuthLevel(pLevel);
	if(Level == -1)
	{
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "level can be one of {\"admin\", \"mod(erator)\", \"helper\"}");
		return;
	}

	MD5_DIGEST Hash;
	unsigned char aSalt[SALT_BYTES];

	if(md5_from_str(&Hash, pPw))
	{
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "Malformed password hash");
		return;
	}
	if(str_hex_decode(aSalt, sizeof(aSalt), pSalt))
	{
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "Malformed salt hash");
		return;
	}

	pManager->UpdateKeyHash(KeySlot, Hash, aSalt, Level);
	pThis->LogoutKey(KeySlot, "key update");

	pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "key updated");
}

void CServer::ConAuthRemove(IConsole::IResult *pResult, void *pUser)
{
	CServer *pThis = (CServer *)pUser;
	CAuthManager *pManager = &pThis->m_AuthManager;

	const char *pIdent = pResult->GetString(0);

	int KeySlot = pManager->FindKey(pIdent);
	if(KeySlot == -1)
	{
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "ident couldn't be found");
		return;
	}

	pThis->AuthRemoveKey(KeySlot);

	if(!pManager->NumNonDefaultKeys())
		pThis->SendRconType(-1, false);

	pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", "key removed, all users logged out");
}

static void ListKeysCallback(const char *pIdent, int Level, void *pUser)
{
	static const char LSTRING[][10] = {"helper", "moderator", "admin"};

	char aBuf[256];
	str_format(aBuf, sizeof(aBuf), "%s %s", pIdent, LSTRING[Level - 1]);
	((CServer *)pUser)->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "auth", aBuf);
}

void CServer::ConAuthList(IConsole::IResult *pResult, void *pUser)
{
	CServer *pThis = (CServer *)pUser;
	CAuthManager *pManager = &pThis->m_AuthManager;

	pManager->ListKeys(ListKeysCallback, pThis);
}

void CServer::ConNameBan(IConsole::IResult *pResult, void *pUser)
{
	CServer *pThis = (CServer *)pUser;
	char aBuf[256];
	const char *pName = pResult->GetString(0);
	const char *pReason = pResult->NumArguments() > 3 ? pResult->GetString(3) : "";
	int Distance = pResult->NumArguments() > 1 ? pResult->GetInteger(1) : str_length(pName) / 3;
	int IsSubstring = pResult->NumArguments() > 2 ? pResult->GetInteger(2) : 0;

	for(auto &Ban : pThis->m_vNameBans)
	{
		if(str_comp(Ban.m_aName, pName) == 0)
		{
			str_format(aBuf, sizeof(aBuf), "changed name='%s' distance=%d old_distance=%d is_substring=%d old_is_substring=%d reason='%s' old_reason='%s'", pName, Distance, Ban.m_Distance, IsSubstring, Ban.m_IsSubstring, pReason, Ban.m_aReason);
			pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "name_ban", aBuf);
			Ban.m_Distance = Distance;
			Ban.m_IsSubstring = IsSubstring;
			str_copy(Ban.m_aReason, pReason);
			return;
		}
	}

	pThis->m_vNameBans.emplace_back(pName, Distance, IsSubstring, pReason);
	str_format(aBuf, sizeof(aBuf), "added name='%s' distance=%d is_substring=%d reason='%s'", pName, Distance, IsSubstring, pReason);
	pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "name_ban", aBuf);
}

void CServer::ConNameUnban(IConsole::IResult *pResult, void *pUser)
{
	CServer *pThis = (CServer *)pUser;
	const char *pName = pResult->GetString(0);

	for(size_t i = 0; i < pThis->m_vNameBans.size(); i++)
	{
		CNameBan *pBan = &pThis->m_vNameBans[i];
		if(str_comp(pBan->m_aName, pName) == 0)
		{
			char aBuf[128];
			str_format(aBuf, sizeof(aBuf), "removed name='%s' distance=%d is_substring=%d reason='%s'", pBan->m_aName, pBan->m_Distance, pBan->m_IsSubstring, pBan->m_aReason);
			pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "name_ban", aBuf);
			pThis->m_vNameBans.erase(pThis->m_vNameBans.begin() + i);
		}
	}
}

void CServer::ConNameBans(IConsole::IResult *pResult, void *pUser)
{
	CServer *pThis = (CServer *)pUser;

	for(auto &Ban : pThis->m_vNameBans)
	{
		char aBuf[128];
		str_format(aBuf, sizeof(aBuf), "name='%s' distance=%d is_substring=%d reason='%s'", Ban.m_aName, Ban.m_Distance, Ban.m_IsSubstring, Ban.m_aReason);
		pThis->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "name_ban", aBuf);
	}
}

void CServer::ConShutdown(IConsole::IResult *pResult, void *pUser)
{
	CServer *pThis = static_cast<CServer *>(pUser);
	pThis->m_RunServer = STOPPING;
	const char *pReason = pResult->GetString(0);
	if(pReason[0])
	{
		str_copy(pThis->m_aShutdownReason, pReason);
	}
}

void CServer::DemoRecorder_HandleAutoStart()
{
	if(Config()->m_SvAutoDemoRecord)
	{
		m_aDemoRecorder[MAX_CLIENTS].Stop();
		char aFilename[IO_MAX_PATH_LENGTH];
		char aDate[20];
		str_timestamp(aDate, sizeof(aDate));
		str_format(aFilename, sizeof(aFilename), "demos/%s_%s.demo", "auto/autorecord", aDate);
		m_aDemoRecorder[MAX_CLIENTS].Start(Storage(), m_pConsole, aFilename, GameServer()->NetVersion(), m_aCurrentMap, &m_aCurrentMapSha256[MAP_TYPE_SIX], m_aCurrentMapCrc[MAP_TYPE_SIX], "server", m_aCurrentMapSize[MAP_TYPE_SIX], m_apCurrentMapData[MAP_TYPE_SIX]);
		if(Config()->m_SvAutoDemoMax)
		{
			// clean up auto recorded demos
			CFileCollection AutoDemos;
			AutoDemos.Init(Storage(), "demos/server", "autorecord", ".demo", Config()->m_SvAutoDemoMax);
		}
	}
}

void CServer::SaveDemo(int ClientID, float Time)
{
	if(IsRecording(ClientID))
	{
		m_aDemoRecorder[ClientID].Stop();

		// rename the demo
		char aOldFilename[IO_MAX_PATH_LENGTH];
		char aNewFilename[IO_MAX_PATH_LENGTH];
		str_format(aOldFilename, sizeof(aOldFilename), "demos/%s_%d_%d_tmp.demo", m_aCurrentMap, m_NetServer.Address().port, ClientID);
		str_format(aNewFilename, sizeof(aNewFilename), "demos/%s_%s_%05.2f.demo", m_aCurrentMap, m_aClients[ClientID].m_aName, Time);
		Storage()->RenameFile(aOldFilename, aNewFilename, IStorage::TYPE_SAVE);
	}
}

void CServer::StartRecord(int ClientID)
{
	if(Config()->m_SvPlayerDemoRecord)
	{
		char aFilename[IO_MAX_PATH_LENGTH];
		str_format(aFilename, sizeof(aFilename), "demos/%s_%d_%d_tmp.demo", m_aCurrentMap, m_NetServer.Address().port, ClientID);
		m_aDemoRecorder[ClientID].Start(Storage(), Console(), aFilename, GameServer()->NetVersion(), m_aCurrentMap, &m_aCurrentMapSha256[MAP_TYPE_SIX], m_aCurrentMapCrc[MAP_TYPE_SIX], "server", m_aCurrentMapSize[MAP_TYPE_SIX], m_apCurrentMapData[MAP_TYPE_SIX]);
	}
}

void CServer::StopRecord(int ClientID)
{
	if(IsRecording(ClientID))
	{
		m_aDemoRecorder[ClientID].Stop();

		char aFilename[IO_MAX_PATH_LENGTH];
		str_format(aFilename, sizeof(aFilename), "demos/%s_%d_%d_tmp.demo", m_aCurrentMap, m_NetServer.Address().port, ClientID);
		Storage()->RemoveFile(aFilename, IStorage::TYPE_SAVE);
	}
}

bool CServer::IsRecording(int ClientID)
{
	return m_aDemoRecorder[ClientID].IsRecording();
}

void CServer::ConRecord(IConsole::IResult *pResult, void *pUser)
{
	CServer *pServer = (CServer *)pUser;
	char aFilename[IO_MAX_PATH_LENGTH];

	if(pResult->NumArguments())
		str_format(aFilename, sizeof(aFilename), "demos/%s.demo", pResult->GetString(0));
	else
	{
		char aDate[20];
		str_timestamp(aDate, sizeof(aDate));
		str_format(aFilename, sizeof(aFilename), "demos/demo_%s.demo", aDate);
	}
	pServer->m_aDemoRecorder[MAX_CLIENTS].Start(pServer->Storage(), pServer->Console(), aFilename, pServer->GameServer()->NetVersion(), pServer->m_aCurrentMap, &pServer->m_aCurrentMapSha256[MAP_TYPE_SIX], pServer->m_aCurrentMapCrc[MAP_TYPE_SIX], "server", pServer->m_aCurrentMapSize[MAP_TYPE_SIX], pServer->m_apCurrentMapData[MAP_TYPE_SIX]);
}

void CServer::ConStopRecord(IConsole::IResult *pResult, void *pUser)
{
	((CServer *)pUser)->m_aDemoRecorder[MAX_CLIENTS].Stop();
}

void CServer::ConMapReload(IConsole::IResult *pResult, void *pUser)
{
	((CServer *)pUser)->m_MapReload = true;
}

void CServer::ConLogout(IConsole::IResult *pResult, void *pUser)
{
	CServer *pServer = (CServer *)pUser;

	if(pServer->m_RconClientID >= 0 && pServer->m_RconClientID < MAX_CLIENTS &&
		pServer->m_aClients[pServer->m_RconClientID].m_State != CServer::CClient::STATE_EMPTY)
	{
		pServer->LogoutClient(pServer->m_RconClientID, "");
	}
}

void CServer::ConShowIps(IConsole::IResult *pResult, void *pUser)
{
	CServer *pServer = (CServer *)pUser;

	if(pServer->m_RconClientID >= 0 && pServer->m_RconClientID < MAX_CLIENTS &&
		pServer->m_aClients[pServer->m_RconClientID].m_State != CServer::CClient::STATE_EMPTY)
	{
		if(pResult->NumArguments())
		{
			pServer->m_aClients[pServer->m_RconClientID].m_ShowIps = pResult->GetInteger(0);
		}
		else
		{
			char aStr[9];
			str_format(aStr, sizeof(aStr), "Value: %d", pServer->m_aClients[pServer->m_RconClientID].m_ShowIps);
			char aBuf[32];
			pServer->SendRconLine(pServer->m_RconClientID, pServer->Console()->Format(aBuf, sizeof(aBuf), "server", aStr));
		}
	}
}

void CServer::ConAddSqlServer(IConsole::IResult *pResult, void *pUserData)
{
	CServer *pSelf = (CServer *)pUserData;

	if(!MysqlAvailable())
	{
		pSelf->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", "can't add MySQL server: compiled without MySQL support");
		return;
	}

	if(!pSelf->Config()->m_SvUseSQL)
		return;

	if(pResult->NumArguments() != 7 && pResult->NumArguments() != 8)
	{
		pSelf->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", "7 or 8 arguments are required");
		return;
	}

	CMysqlConfig Config;
	bool Write;
	if(str_comp_nocase(pResult->GetString(0), "w") == 0)
		Write = false;
	else if(str_comp_nocase(pResult->GetString(0), "r") == 0)
		Write = true;
	else
	{
		pSelf->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", "choose either 'r' for SqlReadServer or 'w' for SqlWriteServer");
		return;
	}

	str_copy(Config.m_aDatabase, pResult->GetString(1), sizeof(Config.m_aDatabase));
	str_copy(Config.m_aPrefix, pResult->GetString(2), sizeof(Config.m_aPrefix));
	str_copy(Config.m_aUser, pResult->GetString(3), sizeof(Config.m_aUser));
	str_copy(Config.m_aPass, pResult->GetString(4), sizeof(Config.m_aPass));
	str_copy(Config.m_aIp, pResult->GetString(5), sizeof(Config.m_aIp));
	str_copy(Config.m_aBindaddr, Config.m_aBindaddr, sizeof(Config.m_aBindaddr));
	Config.m_Port = pResult->GetInteger(6);
	Config.m_Setup = pResult->NumArguments() == 8 ? pResult->GetInteger(7) : true;

	char aBuf[512];
	str_format(aBuf, sizeof(aBuf),
		"Adding new Sql%sServer: DB: '%s' Prefix: '%s' User: '%s' IP: <{%s}> Port: %d",
		Write ? "Write" : "Read",
		Config.m_aDatabase, Config.m_aPrefix, Config.m_aUser, Config.m_aIp, Config.m_Port);
	pSelf->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", aBuf);
	pSelf->DbPool()->RegisterMysqlDatabase(Write ? CDbConnectionPool::WRITE : CDbConnectionPool::READ, &Config);
}

void CServer::ConDumpSqlServers(IConsole::IResult *pResult, void *pUserData)
{
	CServer *pSelf = (CServer *)pUserData;

	if(str_comp_nocase(pResult->GetString(0), "w") == 0)
	{
		pSelf->DbPool()->Print(pSelf->Console(), CDbConnectionPool::WRITE);
		pSelf->DbPool()->Print(pSelf->Console(), CDbConnectionPool::WRITE_BACKUP);
	}
	else if(str_comp_nocase(pResult->GetString(0), "r") == 0)
	{
		pSelf->DbPool()->Print(pSelf->Console(), CDbConnectionPool::READ);
	}
	else
	{
		pSelf->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", "choose either 'r' for SqlReadServer or 'w' for SqlWriteServer");
		return;
	}
}

void CServer::ConchainLoglevel(IConsole::IResult *pResult, void *pUserData, IConsole::FCommandCallback pfnCallback, void *pCallbackUserData)
{
	pfnCallback(pResult, pCallbackUserData);
	log_set_loglevel((LEVEL)g_Config.m_Loglevel);
}

void CServer::ConchainSpecialInfoupdate(IConsole::IResult *pResult, void *pUserData, IConsole::FCommandCallback pfnCallback, void *pCallbackUserData)
{
	pfnCallback(pResult, pCallbackUserData);
	if(pResult->NumArguments())
	{
		CServer *pThis = static_cast<CServer *>(pUserData);
		str_clean_whitespaces(pThis->Config()->m_SvName);
		pThis->UpdateServerInfo(true);
	}
}

void CServer::ConchainMaxclientsperipUpdate(IConsole::IResult *pResult, void *pUserData, IConsole::FCommandCallback pfnCallback, void *pCallbackUserData)
{
	pfnCallback(pResult, pCallbackUserData);
	if(pResult->NumArguments())
		((CServer *)pUserData)->m_NetServer.SetMaxClientsPerIP(pResult->GetInteger(0));
}

void CServer::ConchainCommandAccessUpdate(IConsole::IResult *pResult, void *pUserData, IConsole::FCommandCallback pfnCallback, void *pCallbackUserData)
{
	if(pResult->NumArguments() == 2)
	{
		CServer *pThis = static_cast<CServer *>(pUserData);
		const IConsole::CCommandInfo *pInfo = pThis->Console()->GetCommandInfo(pResult->GetString(0), CFGFLAG_SERVER, false);
		int OldAccessLevel = 0;
		if(pInfo)
			OldAccessLevel = pInfo->GetAccessLevel();
		pfnCallback(pResult, pCallbackUserData);
		if(pInfo && OldAccessLevel != pInfo->GetAccessLevel())
		{
			for(int i = 0; i < MAX_CLIENTS; ++i)
			{
				if(pThis->m_aClients[i].m_State == CServer::CClient::STATE_EMPTY ||
					(pInfo->GetAccessLevel() > AUTHED_ADMIN - pThis->m_aClients[i].m_Authed && AUTHED_ADMIN - pThis->m_aClients[i].m_Authed < OldAccessLevel) ||
					(pInfo->GetAccessLevel() < AUTHED_ADMIN - pThis->m_aClients[i].m_Authed && AUTHED_ADMIN - pThis->m_aClients[i].m_Authed > OldAccessLevel) ||
					(pThis->m_aClients[i].m_pRconCmdToSend && str_comp(pResult->GetString(0), pThis->m_aClients[i].m_pRconCmdToSend->m_pName) >= 0))
					continue;

				if(OldAccessLevel < pInfo->GetAccessLevel())
					pThis->SendRconCmdAdd(pInfo, i);
				else
					pThis->SendRconCmdRem(pInfo, i);
			}
		}
	}
	else
		pfnCallback(pResult, pCallbackUserData);
}

void CServer::LogoutClient(int ClientID, const char *pReason)
{
	if(!IsSixup(ClientID))
	{
		CMsgPacker Msg(NETMSG_RCON_AUTH_STATUS, true);
		Msg.AddInt(0); //authed
		Msg.AddInt(0); //cmdlist
		SendMsg(&Msg, MSGFLAG_VITAL, ClientID);
	}
	else
	{
		CMsgPacker Msg(12, true, true); //NETMSG_RCON_AUTH_OFF
		SendMsg(&Msg, MSGFLAG_VITAL, ClientID);
	}

	m_aClients[ClientID].m_AuthTries = 0;
	m_aClients[ClientID].m_pRconCmdToSend = 0;

	char aBuf[64];
	if(*pReason)
	{
		str_format(aBuf, sizeof(aBuf), "Logged out by %s.", pReason);
		SendRconLine(ClientID, aBuf);
		str_format(aBuf, sizeof(aBuf), "ClientID=%d with key=%s logged out by %s", ClientID, m_AuthManager.KeyIdent(m_aClients[ClientID].m_AuthKey), pReason);
	}
	else
	{
		SendRconLine(ClientID, "Logout successful.");
		str_format(aBuf, sizeof(aBuf), "ClientID=%d with key=%s logged out", ClientID, m_AuthManager.KeyIdent(m_aClients[ClientID].m_AuthKey));
	}

	m_aClients[ClientID].m_Authed = AUTHED_NO;
	m_aClients[ClientID].m_AuthKey = -1;

	GameServer()->OnSetAuthed(ClientID, AUTHED_NO);

	Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", aBuf);
}

void CServer::LogoutKey(int Key, const char *pReason)
{
	for(int i = 0; i < MAX_CLIENTS; i++)
		if(m_aClients[i].m_AuthKey == Key)
			LogoutClient(i, pReason);
}

void CServer::ConchainRconPasswordChangeGeneric(int Level, const char *pCurrent, IConsole::IResult *pResult)
{
	if(pResult->NumArguments() == 1)
	{
		int KeySlot = m_AuthManager.DefaultKey(Level);
		const char *pNew = pResult->GetString(0);
		if(str_comp(pCurrent, pNew) == 0)
		{
			return;
		}
		if(KeySlot == -1 && pNew[0])
		{
			m_AuthManager.AddDefaultKey(Level, pNew);
		}
		else if(KeySlot >= 0)
		{
			if(!pNew[0])
			{
				AuthRemoveKey(KeySlot);
				// Already logs users out.
			}
			else
			{
				m_AuthManager.UpdateKey(KeySlot, pNew, Level);
				LogoutKey(KeySlot, "key update");
			}
		}
	}
}

void CServer::ConchainRconPasswordChange(IConsole::IResult *pResult, void *pUserData, IConsole::FCommandCallback pfnCallback, void *pCallbackUserData)
{
	CServer *pThis = static_cast<CServer *>(pUserData);
	pThis->ConchainRconPasswordChangeGeneric(AUTHED_ADMIN, pThis->Config()->m_SvRconPassword, pResult);
	pfnCallback(pResult, pCallbackUserData);
}

void CServer::ConchainRconModPasswordChange(IConsole::IResult *pResult, void *pUserData, IConsole::FCommandCallback pfnCallback, void *pCallbackUserData)
{
	CServer *pThis = static_cast<CServer *>(pUserData);
	pThis->ConchainRconPasswordChangeGeneric(AUTHED_MOD, pThis->Config()->m_SvRconModPassword, pResult);
	pfnCallback(pResult, pCallbackUserData);
}

void CServer::ConchainRconHelperPasswordChange(IConsole::IResult *pResult, void *pUserData, IConsole::FCommandCallback pfnCallback, void *pCallbackUserData)
{
	CServer *pThis = static_cast<CServer *>(pUserData);
	pThis->ConchainRconPasswordChangeGeneric(AUTHED_HELPER, pThis->Config()->m_SvRconHelperPassword, pResult);
	pfnCallback(pResult, pCallbackUserData);
}

void CServer::ConchainMapUpdate(IConsole::IResult *pResult, void *pUserData, IConsole::FCommandCallback pfnCallback, void *pCallbackUserData)
{
	pfnCallback(pResult, pCallbackUserData);
	if(pResult->NumArguments() >= 1)
	{
		CServer *pThis = static_cast<CServer *>(pUserData);
		pThis->m_MapReload = str_comp(pThis->Config()->m_SvMap, pThis->m_aCurrentMap) != 0;
	}
}

void CServer::ConchainSixupUpdate(IConsole::IResult *pResult, void *pUserData, IConsole::FCommandCallback pfnCallback, void *pCallbackUserData)
{
	pfnCallback(pResult, pCallbackUserData);
	CServer *pThis = static_cast<CServer *>(pUserData);
	if(pResult->NumArguments() >= 1 && pThis->m_aCurrentMap[0] != '\0')
		pThis->m_MapReload |= (pThis->m_apCurrentMapData[MAP_TYPE_SIXUP] != 0) != (pResult->GetInteger(0) != 0);
}

#if defined(CONF_FAMILY_UNIX)
void CServer::ConchainConnLoggingServerChange(IConsole::IResult *pResult, void *pUserData, IConsole::FCommandCallback pfnCallback, void *pCallbackUserData)
{
	pfnCallback(pResult, pCallbackUserData);
	if(pResult->NumArguments() == 1)
	{
		CServer *pServer = (CServer *)pUserData;

		// open socket to send new connections
		if(!pServer->m_ConnLoggingSocketCreated)
		{
			pServer->m_ConnLoggingSocket = net_unix_create_unnamed();
			if(pServer->m_ConnLoggingSocket == -1)
			{
				pServer->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "server", "Failed to created socket for communication with the connection logging server.");
			}
			else
			{
				pServer->m_ConnLoggingSocketCreated = true;
			}
		}

		// set the destination address for the connection logging
		net_unix_set_addr(&pServer->m_ConnLoggingDestAddr, pResult->GetString(0));
	}
}
#endif

void CServer::RegisterCommands()
{
	m_pConsole = Kernel()->RequestInterface<IConsole>();
	m_pGameServer = Kernel()->RequestInterface<IGameServer>();
	m_pMap = Kernel()->RequestInterface<IEngineMap>();
	m_pStorage = Kernel()->RequestInterface<IStorage>();
	m_pAntibot = Kernel()->RequestInterface<IEngineAntibot>();

	HttpInit(m_pStorage);

	// register console commands
	Console()->Register("kick", "i[id] ?r[reason]", CFGFLAG_SERVER, ConKick, this, "Kick player with specified id for any reason");
	Console()->Register("status", "?r[name]", CFGFLAG_SERVER, ConStatus, this, "List players containing name or all players");
	Console()->Register("shutdown", "?r[reason]", CFGFLAG_SERVER, ConShutdown, this, "Shut down");
	Console()->Register("logout", "", CFGFLAG_SERVER, ConLogout, this, "Logout of rcon");
	Console()->Register("show_ips", "?i[show]", CFGFLAG_SERVER, ConShowIps, this, "Show IP addresses in rcon commands (1 = on, 0 = off)");

	Console()->Register("record", "?s[file]", CFGFLAG_SERVER | CFGFLAG_STORE, ConRecord, this, "Record to a file");
	Console()->Register("stoprecord", "", CFGFLAG_SERVER, ConStopRecord, this, "Stop recording");

	Console()->Register("reload", "", CFGFLAG_SERVER, ConMapReload, this, "Reload the map");

	Console()->Register("add_sqlserver", "s['r'|'w'] s[Database] s[Prefix] s[User] s[Password] s[IP] i[Port] ?i[SetUpDatabase ?]", CFGFLAG_SERVER | CFGFLAG_NONTEEHISTORIC, ConAddSqlServer, this, "add a sqlserver");
	Console()->Register("dump_sqlservers", "s['r'|'w']", CFGFLAG_SERVER, ConDumpSqlServers, this, "dumps all sqlservers readservers = r, writeservers = w");

	Console()->Register("auth_add", "s[ident] s[level] r[pw]", CFGFLAG_SERVER | CFGFLAG_NONTEEHISTORIC, ConAuthAdd, this, "Add a rcon key");
	Console()->Register("auth_add_p", "s[ident] s[level] s[hash] s[salt]", CFGFLAG_SERVER | CFGFLAG_NONTEEHISTORIC, ConAuthAddHashed, this, "Add a prehashed rcon key");
	Console()->Register("auth_change", "s[ident] s[level] r[pw]", CFGFLAG_SERVER | CFGFLAG_NONTEEHISTORIC, ConAuthUpdate, this, "Update a rcon key");
	Console()->Register("auth_change_p", "s[ident] s[level] s[hash] s[salt]", CFGFLAG_SERVER | CFGFLAG_NONTEEHISTORIC, ConAuthUpdateHashed, this, "Update a rcon key with prehashed data");
	Console()->Register("auth_remove", "s[ident]", CFGFLAG_SERVER | CFGFLAG_NONTEEHISTORIC, ConAuthRemove, this, "Remove a rcon key");
	Console()->Register("auth_list", "", CFGFLAG_SERVER, ConAuthList, this, "List all rcon keys");

	Console()->Register("name_ban", "s[name] ?i[distance] ?i[is_substring] ?r[reason]", CFGFLAG_SERVER, ConNameBan, this, "Ban a certain nickname");
	Console()->Register("name_unban", "s[name]", CFGFLAG_SERVER, ConNameUnban, this, "Unban a certain nickname");
	Console()->Register("name_bans", "", CFGFLAG_SERVER, ConNameBans, this, "List all name bans");

	RustVersionRegister(*Console());

	Console()->Chain("sv_name", ConchainSpecialInfoupdate, this);
	Console()->Chain("loglevel", ConchainLoglevel, this);
	Console()->Chain("password", ConchainSpecialInfoupdate, this);

	Console()->Chain("sv_max_clients_per_ip", ConchainMaxclientsperipUpdate, this);
	Console()->Chain("access_level", ConchainCommandAccessUpdate, this);

	Console()->Chain("sv_rcon_password", ConchainRconPasswordChange, this);
	Console()->Chain("sv_rcon_mod_password", ConchainRconModPasswordChange, this);
	Console()->Chain("sv_rcon_helper_password", ConchainRconHelperPasswordChange, this);
	Console()->Chain("sv_map", ConchainMapUpdate, this);
	Console()->Chain("sv_sixup", ConchainSixupUpdate, this);

#if defined(CONF_FAMILY_UNIX)
	Console()->Chain("sv_conn_logging_server", ConchainConnLoggingServerChange, this);
#endif

	// register console commands in sub parts
	m_ServerBan.InitServerBan(Console(), Storage(), this);
	m_pGameServer->OnConsoleInit();
}

int CServer::SnapNewID()
{
	return m_IDPool.NewID();
}

void CServer::SnapFreeID(int ID)
{
	m_IDPool.FreeID(ID);
}

void *CServer::SnapNewItem(int Type, int ID, int Size)
{
	dbg_assert(ID >= -1 && ID <= 0xffff, "incorrect id");
	return ID < 0 ? 0 : m_SnapshotBuilder.NewItem(Type, ID, Size);
}

void CServer::SnapSetStaticsize(int ItemType, int Size)
{
	m_SnapshotDelta.SetStaticsize(ItemType, Size);
}

CServer *CreateServer() { return new CServer(); }

// DDRace

void CServer::GetClientAddr(int ClientID, NETADDR *pAddr) const
{
	if(ClientID >= 0 && ClientID < MAX_CLIENTS && m_aClients[ClientID].m_State == CClient::STATE_INGAME)
	{
		*pAddr = *m_NetServer.ClientAddr(ClientID);
	}
}

const char *CServer::GetAnnouncementLine(char const *pFileName)
{
	IOHANDLE File = m_pStorage->OpenFile(pFileName, IOFLAG_READ | IOFLAG_SKIP_BOM, IStorage::TYPE_ALL);
	if(!File)
		return 0;

	std::vector<char *> vpLines;
	char *pLine;
	CLineReader Reader;
	Reader.Init(File);
	while((pLine = Reader.Get()))
		if(str_length(pLine))
			if(pLine[0] != '#')
				vpLines.push_back(pLine);

	if(vpLines.empty())
	{
		return 0;
	}
	else if(vpLines.size() == 1)
	{
		m_AnnouncementLastLine = 0;
	}
	else if(!Config()->m_SvAnnouncementRandom)
	{
		if(++m_AnnouncementLastLine >= vpLines.size())
			m_AnnouncementLastLine %= vpLines.size();
	}
	else
	{
		unsigned Rand;
		do
		{
			Rand = rand() % vpLines.size();
		} while(Rand == m_AnnouncementLastLine);

		m_AnnouncementLastLine = Rand;
	}

	io_close(File);

	return vpLines[m_AnnouncementLastLine];
}

int *CServer::GetIdMap(int ClientID)
{
	return m_aIdMap + VANILLA_MAX_CLIENTS * ClientID;
}

bool CServer::SetTimedOut(int ClientID, int OrigID)
{
	if(!m_NetServer.SetTimedOut(ClientID, OrigID))
	{
		return false;
	}
	m_aClients[ClientID].m_Sixup = m_aClients[OrigID].m_Sixup;

	if(m_aClients[OrigID].m_Authed != AUTHED_NO)
	{
		LogoutClient(ClientID, "Timeout Protection");
	}
	DelClientCallback(OrigID, "Timeout Protection used", this);
	m_aClients[ClientID].m_Authed = AUTHED_NO;
	m_aClients[ClientID].m_Flags = m_aClients[OrigID].m_Flags;
	m_aClients[ClientID].m_DDNetVersion = m_aClients[OrigID].m_DDNetVersion;
	m_aClients[ClientID].m_GotDDNetVersionPacket = m_aClients[OrigID].m_GotDDNetVersionPacket;
	m_aClients[ClientID].m_DDNetVersionSettled = m_aClients[OrigID].m_DDNetVersionSettled;
	return true;
}

void CServer::SetErrorShutdown(const char *pReason)
{
	str_copy(m_aErrorShutdownReason, pReason);
}
