#ifndef ASTAR_H
#define ASTAR_H

#include <algorithm>
#include <cmath>
#include <map>
#include <queue>
#include <unordered_map>
#include <vector>

class AStar
{
public:
	AStar(const std::vector<std::vector<int>> &grid, std::pair<int, int> goal) :
		grid(grid), rows(grid.size()), cols(grid[0].size()), goal(goal)
	{
		distance = std::vector<std::vector<double>>(rows, std::vector<double>(cols, std::numeric_limits<double>::infinity()));
		dijkstra(goal.second, goal.first);
	}

	struct Node
	{
		int x, y;
		double dist;

		Node(int x, int y, double dist) :
			x(x), y(y), dist(dist) {}

		double f() const { return dist; }

		bool operator>(const Node &other) const { return f() > other.f(); }
	};

	int distanceToGoal(int startX,int startY)
	{
		return distance[startY][startX];
	}

	std::vector<std::pair<int, int>> findPath(std::pair<int, int> start, int max_length = 30)
	{
		auto [sx, sy] = start;

		if(!isValid(sx, sy) || start == goal)
		{
			return {}; // Start is invalid
		}
		std::vector<std::pair<int, int>> ret;
		std::vector<std::pair<int, int>> directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
		int x = start.second;
		int y = start.first;
		for(size_t i = 0; i < max_length; i++)
		{
			std::pair<int, int> bestMove = {-1, -1};
			double minDistance = std::numeric_limits<double>::infinity();
			for(const auto &[dx, dy] : directions)
			{
				int nx = x + dx;
				int ny = y + dy;

				if(isValid(nx, ny) && distance[nx][ny] < minDistance)
				{
					minDistance = distance[nx][ny];
					bestMove = {dx, dy};
				}
			}

			if(bestMove == std::pair<int, int>{-1, -1})
			{
				break; // No valid move found, end the search
			}

			auto [dx, dy] = bestMove;
			x += dx;
			y += dy;
			ret.push_back(bestMove);

			if(x == goal.second && y == goal.first)
				break;
		}

		// Path found
		return ret;
	}

private:
	const std::vector<std::vector<int>> &grid;
	int rows, cols;
	std::pair<int, int> goal;
	std::vector<std::vector<double>> distance;

	std::map<std::tuple<std::pair<int, int>, std::pair<int, int>>, std::vector<std::pair<int, int>>> pathCache;

	bool isValid(int y, int x) const
	{
		return y >= 0 && y < rows && x >= 0 && x < cols && grid[y][x] == 0;
	}

	double heuristic(int x1, int y1, int x2, int y2) const
	{
		return std::abs(x1 - x2) + std::abs(y1 - y2); // Manhattan distance
	}

	int getHash(int x, int y) const
	{
		return x * cols + y;
	}

	// Run Dijkstra's algorithm from a single source node
	std::vector<std::vector<double>> dijkstra(int goalY, int goalX)
	{
		std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;

		distance[goalY][goalX] = 0;
		pq.push(Node(goalX, goalY, 0));

		std::vector<std::pair<int, int>> directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

		while(!pq.empty())
		{
			Node current = pq.top();
			pq.pop();

			int x = current.x;
			int y = current.y;
			double dist = current.dist;

			if(dist > distance[y][x])
				continue;

			for(const auto &[dx, dy] : directions)
			{
				int nx = x + dx;
				int ny = y + dy;

				if(isValid(ny, nx) && dist + 1 < distance[ny][nx])
				{
					distance[ny][nx] = dist + 1;
					pq.push(Node(nx, ny, distance[ny][nx]));
				}
			}
		}

		return distance;
	}
};

#endif // ASTAR_H